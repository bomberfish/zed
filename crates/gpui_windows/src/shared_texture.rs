//! Producer-side shared render target for the embed path on Windows.
//!
//! The consumer (slop2's GTK client) imports frames with
//! `GdkD3D12TextureBuilder`, which takes an `ID3D12Resource`. Zed's Windows
//! renderer is D3D11, and a D3D11 shared handle cannot be opened as a D3D12
//! resource — the supported direction is the other way around. So the texture is
//! *created* in D3D12 and projected into D3D11 through the 11on12 interop layer:
//! the renderer draws to it as an ordinary `ID3D11RenderTargetView` while the
//! consumer holds the same pixels as a D3D12 resource.
//!
//! That inverts device creation. `ID3D11On12Device::CreateWrappedResource` only
//! works on a D3D11 device that was itself created by `D3D11On12CreateDevice`
//! on top of a D3D12 device, so an embed session cannot reuse a D3D11 device
//! built the normal way — hence [`SharedTextureDevices`] rather than a helper
//! that takes the existing [`crate::directx_devices::DirectXDevices`].
//!
//! Note this keeps rendering on the *native* driver. The Vulkan export route
//! would have forced the whole UI onto the Mesa-on-D3D12 adapter, since that is
//! the only one here implementing `VK_KHR_external_memory_win32`.

use anyhow::{Context as _, Result};
use windows::Win32::Foundation::{CloseHandle, DUPLICATE_SAME_ACCESS, GENERIC_ALL, HANDLE};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_RENDER_TARGET, D3D11_CREATE_DEVICE_BGRA_SUPPORT, ID3D11Device, ID3D11DeviceContext,
    ID3D11Resource, ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D11on12::{
    D3D11_RESOURCE_FLAGS, D3D11On12CreateDevice, ID3D11On12Device,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_HEAP_FLAG_SHARED,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_DESC,
    D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12CreateDevice,
    ID3D12CommandQueue, ID3D12Device, ID3D12Resource,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::System::Threading::GetCurrentProcess;
use windows::core::Interface;

/// A D3D12 device plus the D3D11 device projected onto it. The renderer uses
/// `device`/`device_context` exactly as it would a normally-created pair.
pub(crate) struct SharedTextureDevices {
    pub(crate) d3d12_device: ID3D12Device,
    /// 11on12 needs a queue to submit the D3D11 work it translates.
    pub(crate) _command_queue: ID3D12CommandQueue,
    pub(crate) d3d11on12: ID3D11On12Device,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
}

impl SharedTextureDevices {
    pub(crate) fn new() -> Result<Self> {
        // Default adapter: unlike the Vulkan route there is no capability to
        // filter on, since sharing is intrinsic to D3D12 rather than an
        // optional extension.
        let mut d3d12_device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut d3d12_device) }
            .context("D3D12CreateDevice for the embed shared texture")?;
        let d3d12_device = d3d12_device.context("D3D12CreateDevice returned no device")?;

        let command_queue: ID3D12CommandQueue = unsafe {
            d3d12_device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .context("CreateCommandQueue for the embed shared texture")?;

        let queues = [Some(command_queue.cast::<windows::core::IUnknown>()?)];
        let feature_levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11On12CreateDevice(
                &d3d12_device.cast::<windows::core::IUnknown>()?,
                // BGRA_SUPPORT to match what the normal device path requests, so
                // Direct2D/DirectWrite interop behaves the same in embed mode.
                D3D11_CREATE_DEVICE_BGRA_SUPPORT.0,
                Some(&feature_levels),
                Some(&queues),
                0,
                Some(&mut device),
                Some(&mut device_context),
                None,
            )
        }
        .context("D3D11On12CreateDevice")?;

        let device = device.context("D3D11On12CreateDevice returned no D3D11 device")?;
        let device_context =
            device_context.context("D3D11On12CreateDevice returned no device context")?;
        let d3d11on12 = device
            .cast::<ID3D11On12Device>()
            .context("casting the 11on12 device")?;

        Ok(Self {
            d3d12_device,
            _command_queue: command_queue,
            d3d11on12,
            device,
            device_context,
        })
    }
}

/// One shared render target: a D3D12 resource, its D3D11 projection, and the
/// render-target view the renderer draws into.
pub(crate) struct SharedTexture {
    /// Held only to keep the shared allocation alive: the renderer works
    /// through the D3D11 projection, and the consumer through its own handle.
    _resource: ID3D12Resource,
    wrapped: ID3D11Texture2D,
    handle: HANDLE,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl SharedTexture {
    pub(crate) fn new(devices: &SharedTextureDevices, width: u32, height: u32) -> Result<Self> {
        let width = width.max(1);
        let height = height.max(1);

        // BGRA to match the swapchain format the on-screen renderer uses, so the
        // shaders need no variant. SHARED heap flag is what makes the resource
        // exportable at all.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: Default::default(),
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            devices.d3d12_device.CreateCommittedResource(
                &heap_properties,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .context("CreateCommittedResource for the shared embed target")?;
        let resource = resource.context("CreateCommittedResource returned no resource")?;

        let handle = unsafe {
            devices.d3d12_device.CreateSharedHandle(
                &resource,
                None,
                GENERIC_ALL.0,
                None,
                // The handle is duplicated per consumer in `share_with`; this one
                // stays owned by this struct and is closed on drop.
            )
        }
        .context("CreateSharedHandle for the shared embed target")?;

        // The wrapped resource carries the D3D11 side's view of state. In/out
        // state are both RENDER_TARGET: the renderer only ever draws to it, and
        // leaving it there avoids a transition on every acquire/release pair.
        let flags = D3D11_RESOURCE_FLAGS {
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            ..Default::default()
        };
        let mut wrapped: Option<ID3D11Texture2D> = None;
        unsafe {
            devices.d3d11on12.CreateWrappedResource(
                &resource,
                &flags,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                &mut wrapped,
            )
        }
        .context("CreateWrappedResource for the shared embed target")?;
        let wrapped = wrapped.context("CreateWrappedResource returned no texture")?;

        Ok(Self {
            _resource: resource,
            wrapped,
            handle,
            width,
            height,
        })
    }

    /// Hand the D3D11 side control of the texture. Must bracket every frame that
    /// draws into it, or the 11on12 layer will not have flushed the D3D12 state
    /// the consumer reads.
    pub(crate) fn acquire(&self, devices: &SharedTextureDevices) {
        let resources = [Some(self.wrapped.cast::<ID3D11Resource>().unwrap_or_else(
            |_| unreachable!("a wrapped texture is always an ID3D11Resource"),
        ))];
        unsafe { devices.d3d11on12.AcquireWrappedResources(&resources) };
    }

    /// Return the texture to D3D12 and flush, so the consumer sees the frame.
    pub(crate) fn release(&self, devices: &SharedTextureDevices) {
        let resources = [Some(self.wrapped.cast::<ID3D11Resource>().unwrap_or_else(
            |_| unreachable!("a wrapped texture is always an ID3D11Resource"),
        ))];
        unsafe {
            devices.d3d11on12.ReleaseWrappedResources(&resources);
            devices.device_context.Flush();
        }
    }

    /// Duplicate the shared handle into the consumer process.
    ///
    /// An NT handle is process-local, so the raw value means nothing on the
    /// other side of the pipe: sending it undiplicated is a bug that presents as
    /// the consumer failing to open a resource that plainly exists.
    pub(crate) fn share_with(&self, consumer_process: HANDLE) -> Result<isize> {
        let mut duplicate = HANDLE::default();
        unsafe {
            windows::Win32::Foundation::DuplicateHandle(
                GetCurrentProcess(),
                self.handle,
                consumer_process,
                &mut duplicate,
                0,
                false,
                DUPLICATE_SAME_ACCESS,
            )
        }
        .context("duplicating the shared texture handle into the consumer")?;
        Ok(duplicate.0 as isize)
    }

    /// The D3D11 projection of the shared resource — what the renderer binds as
    /// its render target.
    pub(crate) fn render_target_texture(&self) -> &ID3D11Texture2D {
        &self.wrapped
    }
}

impl Drop for SharedTexture {
    fn drop(&mut self) {
        if !self.handle.is_invalid()
            && let Err(error) = unsafe { CloseHandle(self.handle) }
        {
            log::error!("closing the shared embed texture handle: {error}");
        }
    }
}
