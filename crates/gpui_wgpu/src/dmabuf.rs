//! Producer-side dma-buf export for the headless embed path.
//!
//! Renders into a LINEAR-tiled `VkImage` whose memory is exportable as a Linux
//! DRM PRIME (dma-buf) fd, so the consumer (slop2) can import it directly into
//! GTK with no CPU readback. The wgpu render path copies each frame into this
//! image; the fd + plane geometry are handed to the consumer once, and a small
//! "frame ready" notification is sent per frame.
//!
//! This mirrors the standalone `dmabuf_poc` proof, ported onto gpui_wgpu's
//! offscreen renderer.

use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};

use anyhow::{Context as _, Result, bail};
use ash::vk;
use wgpu::hal::api::Vulkan;

pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// DRM fourcc for wgpu `Rgba8Unorm`: `DRM_FORMAT_ABGR8888`, documented as
/// "[31:0] A:B:G:R little endian" — memory byte order R, G, B, A.
pub const DRM_FORMAT_ABGR8888: u32 = fourcc(b'A', b'B', b'2', b'4');

const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

pub type ExportInfo = DmabufInfo;
pub type ExportTarget = DmabufTarget;

/// A DRM PRIME fd. The Windows sibling of this module exports an NT handle
/// instead, so the renderer names the handle type rather than either concretely.
pub type ExportHandle = RawFd;

/// Everything the consumer needs to import one dma-buf plane.
#[derive(Clone, Copy, Debug)]
pub struct DmabufInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub offset: u32,
    pub modifier: u64,
    pub fourcc: u32,
}

/// Create a wgpu device with the Vulkan external-memory extensions needed to
/// export dma-buf fds. wgpu doesn't allow adding arbitrary device extensions
/// through `request_device`, so we open the hal device with a callback that
/// appends them, then hand it back to wgpu.
pub fn create_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let features = wgpu::Features::empty();
    let limits = wgpu::Limits::downlevel_defaults()
        .using_resolution(adapter.limits())
        .using_alignment(adapter.limits());
    let memory_hints = wgpu::MemoryHints::MemoryUsage;

    let open_device = {
        let hal_adapter =
            unsafe { adapter.as_hal::<Vulkan>() }.context("adapter is not Vulkan")?;
        let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback> =
            Box::new(|args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory::NAME);
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
            });
        unsafe { hal_adapter.open_with_callback(features, &limits, &memory_hints, Some(callback)) }
            .context("open Vulkan device with dma-buf extensions")?
    };

    let (device, queue) = unsafe {
        adapter.create_device_from_hal::<Vulkan>(
            open_device,
            &wgpu::DeviceDescriptor {
                label: Some("gpui-dmabuf-device"),
                required_features: features,
                required_limits: limits,
                memory_hints,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        )
    }
    .context("create_device_from_hal")?;
    Ok((device, queue))
}

/// An exportable render target: a wgpu texture backed by dma-buf memory.
pub struct DmabufTarget {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    _fd: OwnedFd,
    pub info: DmabufInfo,
    /// wgpu view of the same VkImage — the copy destination each frame.
    pub texture: wgpu::Texture,
}

impl DmabufTarget {
    pub fn new(
        device: &wgpu::Device,
        instance: &ash::Instance,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let width = width.max(1);
        let height = height.max(1);

        let (ash_device, physical) = {
            let hal_device =
                unsafe { device.as_hal::<Vulkan>() }.context("offscreen device is not Vulkan")?;
            (
                hal_device.raw_device().clone(),
                hal_device.raw_physical_device(),
            )
        };

        // --- exportable LINEAR image --------------------------------------
        let mut external_ci = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_ci);
        let image = unsafe { ash_device.create_image(&image_ci, None) }.context("vkCreateImage")?;

        // --- exportable dedicated memory ----------------------------------
        let reqs = unsafe { ash_device.get_image_memory_requirements(image) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let mem_type = pick_memory_type(&mem_props, reqs.memory_type_bits)
            .context("no suitable memory type for dma-buf image")?;

        let mut export_ci = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut dedicated_ci = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type)
            .push_next(&mut export_ci)
            .push_next(&mut dedicated_ci);
        let memory =
            unsafe { ash_device.allocate_memory(&alloc_ci, None) }.context("vkAllocateMemory")?;
        unsafe { ash_device.bind_image_memory(image, memory, 0) }.context("vkBindImageMemory")?;

        // --- export the dma-buf fd ----------------------------------------
        let external_memory_fd = ash::khr::external_memory_fd::Device::new(instance, &ash_device);
        let raw_fd: RawFd = unsafe {
            external_memory_fd.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            )
        }
        .context("vkGetMemoryFdKHR")?;
        if raw_fd < 0 {
            bail!("vkGetMemoryFdKHR returned invalid fd");
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // --- plane layout -------------------------------------------------
        let layout = unsafe {
            ash_device.get_image_subresource_layout(
                image,
                vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .array_layer(0),
            )
        };

        // --- wrap the VkImage as a wgpu texture (wgpu-hal must not own it) --
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("dmabuf-target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::wgt::TextureUses::COPY_DST,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let hal_texture = {
            let hal_device = unsafe { device.as_hal::<Vulkan>() }
                .context("device not Vulkan when wrapping texture")?;
            unsafe {
                hal_device.texture_from_raw(
                    image,
                    &hal_desc,
                    None,
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            }
        };
        let texture = unsafe {
            device.create_texture_from_hal::<Vulkan>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("dmabuf-target"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )
        };

        Ok(Self {
            device: ash_device,
            image,
            memory,
            _fd: fd,
            info: DmabufInfo {
                width,
                height,
                stride: layout.row_pitch as u32,
                offset: layout.offset as u32,
                modifier: DRM_FORMAT_MOD_LINEAR,
                fourcc: DRM_FORMAT_ABGR8888,
            },
            texture,
        })
    }

    pub fn raw_fd(&self) -> RawFd {
        self._fd.as_raw_fd()
    }

    /// The platform-neutral spelling of [`Self::raw_fd`], so the renderer can
    /// hand out export handles without knowing which OS it is on.
    pub fn export_handle(&self) -> ExportHandle {
        self.raw_fd()
    }
}

impl Drop for DmabufTarget {
    fn drop(&mut self) {
        // The wgpu texture (External memory) won't free the image; we do.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// Prefer device-local memory, else anything matching the requirements.
fn pick_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Option<u32> {
    let types = props.memory_types_as_slice();
    for (i, ty) in types.iter().enumerate() {
        if type_bits & (1 << i) != 0
            && ty
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Some(i as u32);
        }
    }
    (0..types.len())
        .find(|&i| type_bits & (1 << i) != 0)
        .map(|i| i as u32)
}
