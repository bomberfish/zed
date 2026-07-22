// Producer-side dma-buf export: render into a Vulkan image whose memory is
// exportable as a Linux DRM PRIME (dma-buf) fd, so a separate process can import
// it directly into GTK/GL with no CPU readback.
//
// Path: create a LINEAR-tiled `VkImage` with `VkExternalMemoryImageCreateInfo`,
// back it with exportable dedicated `VkDeviceMemory`, export the fd via
// `vkGetMemoryFdKHR`, and wrap the image as a `wgpu::Texture` (via wgpu-hal's
// `texture_from_raw`) so the existing wgpu render path can copy frames into it.
//
// LINEAR tiling + `DRM_FORMAT_MOD_LINEAR` is the most broadly importable layout
// and sidesteps DRM-format-modifier negotiation for this first cut.

use anyhow::{Context as _, Result, bail};
use ash::vk;
use std::os::fd::{FromRawFd as _, OwnedFd, RawFd};

use wgpu::hal::api::Vulkan;

pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// DRM fourcc for wgpu `Rgba8Unorm`. `DRM_FORMAT_ABGR8888` is documented as
/// "[31:0] A:B:G:R little endian", i.e. memory byte order R, G, B, A — matching
/// `Rgba8Unorm`. DRM format codes spell the depth with digits ('2','4'), so the
/// code is `fourcc('A','B','2','4')`, not the literal "ABGR".
pub const DRM_FORMAT_ABGR8888: u32 = fourcc(b'A', b'B', b'2', b'4');

const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

/// Create a wgpu device from this adapter with the Vulkan external-memory
/// extensions needed to export dma-buf fds. wgpu doesn't let you add arbitrary
/// device extensions through `request_device`, so we open the hal device with a
/// callback that appends them, then hand it back to wgpu.
pub fn create_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue)> {
    let features = wgpu::Features::empty();
    let limits = wgpu::Limits::default();
    let memory_hints = wgpu::MemoryHints::default();

    let open_device = {
        let hal_adapter =
            unsafe { adapter.as_hal::<Vulkan>() }.context("adapter is not Vulkan")?;
        let callback: Box<wgpu::hal::vulkan::CreateDeviceCallback> =
            Box::new(|args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory::NAME);
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
            });
        unsafe {
            hal_adapter.open_with_callback(features, &limits, &memory_hints, Some(callback))
        }
        .context("open Vulkan device with dma-buf extensions")?
    };

    let (device, queue) = unsafe {
        adapter.create_device_from_hal::<Vulkan>(
            open_device,
            &wgpu::DeviceDescriptor {
                label: Some("dmabuf-device"),
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

/// An exportable render target: a wgpu texture backed by dma-buf memory.
pub struct DmabufTarget {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    memory_size: u64,
    host_visible: bool,
    /// Kept alive so the fd stays valid; the consumer receives dup'd copies.
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

        // Pull the raw Vulkan handles out of wgpu-hal.
        let (ash_device, physical) = {
            let hal_device =
                unsafe { device.as_hal::<Vulkan>() }.context("offscreen device is not Vulkan")?;
            (hal_device.raw_device().clone(), hal_device.raw_physical_device())
        };

        let format = vk::Format::R8G8B8A8_UNORM;

        // --- exportable image ---------------------------------------------
        let mut external_ci = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
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
        let image =
            unsafe { ash_device.create_image(&image_ci, None) }.context("vkCreateImage")?;

        // --- exportable dedicated memory ----------------------------------
        let reqs = unsafe { ash_device.get_image_memory_requirements(image) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (mem_type, host_visible) = pick_memory_type(&mem_props, reqs.memory_type_bits)
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
        let external_memory_fd =
            ash::khr::external_memory_fd::Device::new(instance, &ash_device);
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

        // --- plane layout (stride/offset) ---------------------------------
        let layout = unsafe {
            ash_device.get_image_subresource_layout(
                image,
                vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .array_layer(0),
            )
        };

        // --- wrap the VkImage as a wgpu texture ---------------------------
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
        // wgpu-hal must NOT own the image/memory — we free them in Drop.
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
            memory_size: reqs.size,
            host_visible,
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
        use std::os::fd::AsRawFd as _;
        self._fd.as_raw_fd()
    }

    /// Read the exported buffer back on the CPU (host-visible memory only) for
    /// headless verification. Returns tightly-packed RGBA.
    pub fn map_and_read(&self) -> Result<Vec<u8>> {
        if !self.host_visible {
            bail!("dma-buf memory is not host-visible; cannot map for verification");
        }
        let ptr = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.memory_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .context("vkMapMemory")? as *const u8;
        let stride = self.info.stride as usize;
        let tight = self.info.width as usize * 4;
        let mut out = vec![0u8; tight * self.info.height as usize];
        for row in 0..self.info.height as usize {
            let src = self.info.offset as usize + row * stride;
            let dst = row * tight;
            let slice = unsafe { std::slice::from_raw_parts(ptr.add(src), tight) };
            out[dst..dst + tight].copy_from_slice(slice);
        }
        unsafe { self.device.unmap_memory(self.memory) };
        Ok(out)
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

/// Prefer memory that's both device-local and host-visible (so the POC can map
/// and verify), then device-local, then anything matching the requirements.
fn pick_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<(u32, bool)> {
    let types = props.memory_types_as_slice();
    let matches = |i: usize, flags: vk::MemoryPropertyFlags| {
        type_bits & (1 << i) != 0 && types[i].property_flags.contains(flags)
    };
    let host = vk::MemoryPropertyFlags::DEVICE_LOCAL
        | vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT;
    for i in 0..types.len() {
        if matches(i, host) {
            return Some((i as u32, true));
        }
    }
    for i in 0..types.len() {
        if matches(i, vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            return Some((i as u32, false));
        }
    }
    (0..types.len())
        .find(|&i| type_bits & (1 << i) != 0)
        .map(|i| (i as u32, false))
}
