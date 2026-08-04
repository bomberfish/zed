//! Producer-side shared-texture export for the headless embed path on Windows.
//!
//! The counterpart to `dmabuf.rs`: renders into a `VkImage` whose memory is
//! exported as a shareable D3D12 resource, so the consumer imports it directly
//! instead of reading pixels back over the CPU. GTK 4.20+ takes exactly that on
//! the other end — `GdkD3D12TextureBuilder` wraps an `ID3D12Resource` (opened
//! from this handle with `ID3D12Device::OpenSharedHandle`) as a `GdkTexture`.
//!
//! Four things differ from the dma-buf side, and each shapes the code here:
//!
//! * The handle type is `D3D12_RESOURCE`, not `OPAQUE_WIN32`. Opaque memory is
//!   only meaningful to another Vulkan device with a matching `deviceUUID`;
//!   GTK's importer is D3D12, so the export has to be a D3D12 resource.
//! * Not every adapter can do this. On this hardware the *native* Qualcomm
//!   Vulkan driver does not implement `VK_KHR_external_memory_win32` at all, and
//!   reports `exportable=false` for every handle type; only the Mesa-on-D3D12
//!   ("Dozen") ICD can export. Both report the same `deviceLUID`, which is what
//!   makes the handle openable on the D3D12 adapter GTK is using — so the device
//!   must be opened on an adapter that advertises the extension rather than on
//!   whichever one `HighPerformance` prefers.
//! * `D3D12_RESOURCE` is only supported with `OPTIMAL` tiling and only as a
//!   dedicated allocation (`LINEAR` reports unsupported). Unlike dma-buf there
//!   is therefore no stride/offset to negotiate: the layout lives in the D3D12
//!   resource and the consumer reads it from there.
//! * An NT handle is process-local: unlike a DRM PRIME fd over `SCM_RIGHTS`, the
//!   value returned here means nothing in the consumer until it is passed through
//!   `DuplicateHandle` into that process. Sending the number alone is a bug.

use anyhow::{Context as _, Result, bail};
use ash::vk;
use wgpu::hal::api::Vulkan;
use windows::Win32::Foundation::{CloseHandle, HANDLE};

/// Everything the consumer needs to import the shared resource. Deliberately
/// smaller than `DmabufInfo`: `OpenSharedHandle` recovers the format and layout
/// from the resource itself, so only the geometry has to travel.
#[derive(Clone, Copy, Debug)]
pub struct SharedTextureInfo {
    pub width: u32,
    pub height: u32,
}

pub type ExportInfo = SharedTextureInfo;
pub type ExportTarget = SharedTarget;

/// A raw NT `HANDLE` as an integer, so it stays `Send` and callers can move it
/// around without depending on `windows` types.
pub type ExportHandle = isize;

/// Pick an adapter that can actually export shared handles.
///
/// Preference by performance is the wrong axis here: on this hardware the
/// fastest adapter (the native Qualcomm driver) is precisely the one that does
/// not implement `VK_KHR_external_memory_win32`, so a zero-copy export on it is
/// impossible and the embed would fall back to CPU readback. Adapters are
/// filtered on the extension instead, and the caller keeps its usual selection
/// when none qualifies.
pub async fn select_exportable_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .find(|adapter| {
            let Some(hal_adapter) = (unsafe { adapter.as_hal::<Vulkan>() }) else {
                return false;
            };
            let raw_instance = hal_adapter.shared_instance().raw_instance();
            let physical_device = hal_adapter.raw_physical_device();
            unsafe { raw_instance.enumerate_device_extension_properties(physical_device) }
                .map(|extensions| {
                    extensions.iter().any(|extension| {
                        extension.extension_name_as_c_str()
                            == Ok(ash::khr::external_memory_win32::NAME)
                    })
                })
                .unwrap_or(false)
        })
}

/// Create a wgpu device with the Vulkan external-memory extensions needed to
/// export NT handles. wgpu doesn't allow adding arbitrary device extensions
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
                args.extensions.push(ash::khr::external_memory_win32::NAME);
            });
        unsafe { hal_adapter.open_with_callback(features, &limits, &memory_hints, Some(callback)) }
            .context("open Vulkan device with external-memory-win32 extensions")?
    };

    let (device, queue) = unsafe {
        adapter.create_device_from_hal::<Vulkan>(
            open_device,
            &wgpu::DeviceDescriptor {
                label: Some("gpui-shared-texture-device"),
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

/// An exportable render target: a wgpu texture backed by shareable memory.
pub struct SharedTarget {
    device: ash::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    handle: ExportHandle,
    pub info: SharedTextureInfo,
    /// wgpu view of the same VkImage — the copy destination each frame.
    pub texture: wgpu::Texture,
}

impl SharedTarget {
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

        // OPTIMAL, and SAMPLED alongside TRANSFER_DST: this is the exact
        // configuration the driver reports as exportable for D3D12_RESOURCE
        // (LINEAR is rejected outright), and the consumer samples the result.
        let mut external_ci = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE);
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
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external_ci);
        let image = unsafe { ash_device.create_image(&image_ci, None) }.context("vkCreateImage")?;

        let requirements = unsafe { ash_device.get_image_memory_requirements(image) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = pick_memory_type(&memory_properties, requirements.memory_type_bits)
            .context("no suitable memory type for shared image")?;

        // `ExportMemoryWin32HandleInfoKHR` is optional for OPAQUE_WIN32 but
        // mandatory for the D3D12 handle types (VUID-VkMemoryAllocateInfo-pNext-00639).
        // Null attributes gives the default security descriptor, which is what we
        // want for a handle duplicated to one known child process; GENERIC_ALL
        // because the consumer both samples the resource and waits on it.
        const GENERIC_ALL: u32 = 0x1000_0000;
        let mut export_handle_ci =
            vk::ExportMemoryWin32HandleInfoKHR::default().dw_access(GENERIC_ALL);
        // Dedicated is not a preference here: the driver reports D3D12_RESOURCE
        // as DEDICATED_ONLY, so the allocation must name the image it backs.
        let mut export_ci = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE);
        let mut dedicated_ci = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_ci = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export_handle_ci)
            .push_next(&mut export_ci)
            .push_next(&mut dedicated_ci);
        let memory =
            unsafe { ash_device.allocate_memory(&alloc_ci, None) }.context("vkAllocateMemory")?;
        unsafe { ash_device.bind_image_memory(image, memory, 0) }.context("vkBindImageMemory")?;

        let external_memory_win32 =
            ash::khr::external_memory_win32::Device::new(instance, &ash_device);
        let handle = unsafe {
            external_memory_win32.get_memory_win32_handle(
                &vk::MemoryGetWin32HandleInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE),
            )
        }
        .context("vkGetMemoryWin32HandleKHR")?;
        // ash types `vk::HANDLE` as a plain `isize` rather than a pointer, so
        // the null check is a comparison rather than `is_null`.
        if handle == 0 {
            bail!("vkGetMemoryWin32HandleKHR returned a null handle");
        }

        // No `vkGetImageSubresourceLayout` here: it is only defined for LINEAR
        // images, and the consumer gets the layout from the D3D12 resource.
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("shared-texture-target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::wgt::TextureUses::COPY_DST | wgpu::wgt::TextureUses::RESOURCE,
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
                    label: Some("shared-texture-target"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };

        Ok(Self {
            device: ash_device,
            image,
            memory,
            handle: handle as ExportHandle,
            info: SharedTextureInfo { width, height },
            texture,
        })
    }

    /// The NT handle for this target's memory, valid in *this* process only.
    /// See the module docs: it must be duplicated into the consumer.
    pub fn export_handle(&self) -> ExportHandle {
        self.handle
    }
}

impl Drop for SharedTarget {
    fn drop(&mut self) {
        // The wgpu texture (External memory) won't free the image; we do.
        unsafe {
            if let Err(error) = self.device.device_wait_idle() {
                log::error!("device_wait_idle before shared-target teardown: {error}");
            }
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
            if let Err(error) = CloseHandle(HANDLE(self.handle as *mut std::ffi::c_void)) {
                log::error!("closing shared-texture handle: {error}");
            }
        }
    }
}

/// Prefer device-local memory, else anything matching the requirements.
fn pick_memory_type(props: &vk::PhysicalDeviceMemoryProperties, type_bits: u32) -> Option<u32> {
    let types = props.memory_types_as_slice();
    for (index, memory_type) in types.iter().enumerate() {
        if type_bits & (1 << index) != 0
            && memory_type
                .property_flags
                .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            return Some(index as u32);
        }
    }
    (0..types.len())
        .find(|&index| type_bits & (1 << index) != 0)
        .map(|index| index as u32)
}
