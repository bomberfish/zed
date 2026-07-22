// Headless wgpu render + IPC + input proof of concept.
//
// Spike for "Zed-in-a-tab": render fully offscreen with wgpu, ship frames to a
// separate consumer process over a unix socket, and take input back the same
// way — no window, no compositor. This binary has three modes:
//
//   dmabuf_poc once      render one frame, read it back, print a pixel (sanity)
//   dmabuf_poc producer  render loop; serve frames + accept input on a socket
//   dmabuf_poc probe     connect to a producer, drive input, verify it landed
//
// Transport today is CPU readback (`copy_texture_to_buffer` → `GdkMemoryTexture`
// on the consumer). The zero-copy dma-buf swap comes next; render + IPC + input
// are identical either way, so proving them first de-risks the rest.

mod dmabuf_export;

use anyhow::{Context as _, Result, bail};
use dmabuf_export::{DmabufInfo, DmabufTarget};
use nix::sys::socket::{
    ControlMessage, ControlMessageOwned, MsgFlags, recvmsg, sendmsg,
};
use std::io::{IoSlice, IoSliceMut, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SOCKET_PATH: &str = "/tmp/dmabuf_poc.sock";

// Triangle centered on a uniform-provided clip-space offset, so the consumer's
// pointer position (forwarded over IPC) visibly moves it.
const SHADER: &str = r#"
struct Uniforms { center: vec2<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.18),
        vec2<f32>(-0.18, -0.18),
        vec2<f32>( 0.18, -0.18),
    );
    return vec4<f32>(p[i] + u.center, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.9, 0.4, 0.1, 1.0);
}
"#;

// ---- IPC protocol -------------------------------------------------------

// Consumer → producer. Coordinates are in texture pixels (top-left origin).
#[derive(Debug, Clone, Copy)]
enum Input {
    PointerMove { x: f32, y: f32 },
    Button { button: u8, pressed: bool },
    Key { keyval: u32, pressed: bool },
    Scroll { dx: f32, dy: f32 },
}

impl Input {
    fn write(&self, w: &mut impl Write) -> std::io::Result<()> {
        match *self {
            Input::PointerMove { x, y } => {
                w.write_all(&[0x01])?;
                w.write_all(&x.to_le_bytes())?;
                w.write_all(&y.to_le_bytes())?;
            }
            Input::Button { button, pressed } => {
                w.write_all(&[0x02, button, pressed as u8])?;
            }
            Input::Key { keyval, pressed } => {
                w.write_all(&[0x03])?;
                w.write_all(&keyval.to_le_bytes())?;
                w.write_all(&[pressed as u8])?;
            }
            Input::Scroll { dx, dy } => {
                w.write_all(&[0x04])?;
                w.write_all(&dx.to_le_bytes())?;
                w.write_all(&dy.to_le_bytes())?;
            }
        }
        w.flush()
    }

    fn read(r: &mut impl Read) -> std::io::Result<Input> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        let f32_from = |r: &mut dyn Read| -> std::io::Result<f32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            Ok(f32::from_le_bytes(b))
        };
        match tag[0] {
            0x01 => Ok(Input::PointerMove {
                x: f32_from(r)?,
                y: f32_from(r)?,
            }),
            0x02 => {
                let mut b = [0u8; 2];
                r.read_exact(&mut b)?;
                Ok(Input::Button {
                    button: b[0],
                    pressed: b[1] != 0,
                })
            }
            0x03 => {
                let mut kv = [0u8; 4];
                r.read_exact(&mut kv)?;
                let mut p = [0u8; 1];
                r.read_exact(&mut p)?;
                Ok(Input::Key {
                    keyval: u32::from_le_bytes(kv),
                    pressed: p[0] != 0,
                })
            }
            0x04 => Ok(Input::Scroll {
                dx: f32_from(r)?,
                dy: f32_from(r)?,
            }),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad input tag {other}"),
            )),
        }
    }
}

// Producer → consumer: b"FRAM" + width + height + stride(bytes/row) + len + RGBA.
fn write_frame(w: &mut impl Write, width: u32, height: u32, stride: u32, data: &[u8]) -> std::io::Result<()> {
    w.write_all(b"FRAM")?;
    w.write_all(&width.to_le_bytes())?;
    w.write_all(&height.to_le_bytes())?;
    w.write_all(&stride.to_le_bytes())?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(data)?;
    w.flush()
}

struct Frame {
    width: u32,
    height: u32,
    stride: u32,
    data: Vec<u8>,
}

fn read_frame(r: &mut impl Read) -> std::io::Result<Frame> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"FRAM" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad frame magic",
        ));
    }
    let mut u = [0u8; 4];
    r.read_exact(&mut u)?;
    let width = u32::from_le_bytes(u);
    r.read_exact(&mut u)?;
    let height = u32::from_le_bytes(u);
    r.read_exact(&mut u)?;
    let stride = u32::from_le_bytes(u);
    r.read_exact(&mut u)?;
    let len = u32::from_le_bytes(u) as usize;
    let mut data = vec![0u8; len];
    r.read_exact(&mut data)?;
    Ok(Frame {
        width,
        height,
        stride,
        data,
    })
}

// ---- Renderer -----------------------------------------------------------

struct Renderer {
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    readback: wgpu::Buffer,
    padded_row: u32,
}

impl Renderer {
    // `dmabuf`: create the device with external-memory extensions so its
    // textures can be exported as dma-buf fds.
    async fn new(dmabuf: bool) -> Result<Self> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .context("no wgpu adapter (need a Vulkan-capable GPU)")?;
        eprintln!("[poc] adapter: {}", adapter.get_info().name);

        let (device, queue) = if dmabuf {
            dmabuf_export::create_device(&adapter)?
        } else {
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("dmabuf_poc device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::Off,
                })
                .await
                .context("request_device failed")?
        };

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("poc shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform"),
            size: 16, // vec2<f32> padded to 16
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("poc pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (WIDTH * 4).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            instance,
            device,
            queue,
            target,
            target_view,
            pipeline,
            uniform,
            bind_group,
            readback,
            padded_row,
        })
    }

    /// The underlying Vulkan instance, needed to export dma-buf memory.
    fn ash_instance(&self) -> ash::Instance {
        unsafe {
            self.instance
                .as_hal::<wgpu::hal::api::Vulkan>()
                .expect("vulkan instance")
                .shared_instance()
                .raw_instance()
                .clone()
        }
    }

    // Render the triangle at pointer `(px, py)` into `target`, then copy it into
    // `dst` (the dma-buf texture). Blocks until the GPU has finished so the
    // exported buffer is stable for the consumer to read.
    fn render_into(&self, px: f32, py: f32, dst: &wgpu::Texture) -> Result<()> {
        let cx = (px / WIDTH as f32) * 2.0 - 1.0;
        let cy = 1.0 - (py / HEIGHT as f32) * 2.0;
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck_cast(&[cx, cy, 0.0, 0.0]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("poc-dmabuf") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("poc pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dst,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .context("poll failed")?;
        Ok(())
    }

    // Render with the triangle centered at pointer `(px, py)` (texture pixels),
    // then read the frame back as tightly-packed RGBA.
    fn render_readback(&self, px: f32, py: f32) -> Result<Vec<u8>> {
        let cx = (px / WIDTH as f32) * 2.0 - 1.0;
        let cy = 1.0 - (py / HEIGHT as f32) * 2.0;
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck_cast(&[cx, cy, 0.0, 0.0]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("poc") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("poc pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .context("poll failed")?;
        rx.recv().context("map channel dropped")?.context("map failed")?;

        // Repack padded rows → tight RGBA.
        let mapped = slice.get_mapped_range();
        let mut out = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
        let tight = (WIDTH * 4) as usize;
        for row in 0..HEIGHT as usize {
            let src = row * self.padded_row as usize;
            let dst = row * tight;
            out[dst..dst + tight].copy_from_slice(&mapped[src..src + tight]);
        }
        drop(mapped);
        self.readback.unmap();
        Ok(out)
    }
}

// Tiny local replacement for bytemuck (avoid the extra dep for one cast).
fn bytemuck_cast(v: &[f32; 4]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 16) }
}

// ---- Modes --------------------------------------------------------------

fn main() -> Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "once".into());
    match mode.as_str() {
        "once" => pollster::block_on(run_once()),
        "producer" => pollster::block_on(run_producer()),
        "probe" => run_probe(),
        "dmabuf-once" => pollster::block_on(run_dmabuf_once()),
        "dmabuf-producer" => pollster::block_on(run_dmabuf_producer()),
        "dmabuf-probe" => run_dmabuf_probe(),
        other => bail!(
            "unknown mode {other:?} (once|producer|probe|dmabuf-once|dmabuf-producer|dmabuf-probe)"
        ),
    }
}

// ---- dma-buf IPC --------------------------------------------------------
//
// Handshake (sent once per connection via SCM_RIGHTS, carrying the plane fds):
//   b"DMAB" + num_fds u32 + width + height + stride + offset + modifier(u64) + fourcc
// Per-frame ready notification (plain write): b"RDY!" + buffer_index u32.

fn send_handshake(socket_fd: RawFd, info: &DmabufInfo, fds: &[RawFd]) -> Result<()> {
    let mut payload = Vec::with_capacity(32);
    payload.extend_from_slice(b"DMAB");
    payload.extend_from_slice(&(fds.len() as u32).to_le_bytes());
    payload.extend_from_slice(&info.width.to_le_bytes());
    payload.extend_from_slice(&info.height.to_le_bytes());
    payload.extend_from_slice(&info.stride.to_le_bytes());
    payload.extend_from_slice(&info.offset.to_le_bytes());
    payload.extend_from_slice(&info.modifier.to_le_bytes());
    payload.extend_from_slice(&info.fourcc.to_le_bytes());

    let iov = [IoSlice::new(&payload)];
    let cmsgs = [ControlMessage::ScmRights(fds)];
    sendmsg::<()>(socket_fd, &iov, &cmsgs, MsgFlags::empty(), None).context("sendmsg handshake")?;
    Ok(())
}

fn recv_handshake(socket_fd: RawFd) -> Result<(DmabufInfo, Vec<OwnedFd>)> {
    let mut buf = [0u8; 64];
    let mut iov = [IoSliceMut::new(&mut buf)];
    let mut cmsg_space = nix::cmsg_space!([RawFd; 4]);
    let msg = recvmsg::<()>(
        socket_fd,
        &mut iov,
        Some(&mut cmsg_space),
        MsgFlags::empty(),
    )
    .context("recvmsg handshake")?;

    let mut fds = Vec::new();
    for cmsg in msg.cmsgs().context("iterate cmsgs")? {
        if let ControlMessageOwned::ScmRights(received) = cmsg {
            for raw in received {
                fds.push(unsafe { OwnedFd::from_raw_fd(raw) });
            }
        }
    }
    if &buf[0..4] != b"DMAB" {
        bail!("bad handshake magic");
    }
    let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    let num_fds = u32_at(4) as usize;
    if fds.len() != num_fds {
        bail!("handshake fd count mismatch: got {}, want {num_fds}", fds.len());
    }
    let info = DmabufInfo {
        width: u32_at(8),
        height: u32_at(12),
        stride: u32_at(16),
        offset: u32_at(20),
        modifier: u64::from_le_bytes([
            buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31],
        ]),
        fourcc: u32_at(32),
    };
    Ok((info, fds))
}

fn send_ready(w: &mut impl Write, index: u32) -> std::io::Result<()> {
    w.write_all(b"RDY!")?;
    w.write_all(&index.to_le_bytes())?;
    // Active width/height (the POC buffers aren't over-allocated, so full size).
    w.write_all(&WIDTH.to_le_bytes())?;
    w.write_all(&HEIGHT.to_le_bytes())?;
    w.flush()
}

async fn run_dmabuf_producer() -> Result<()> {
    let renderer = Renderer::new(true).await?;
    let instance = renderer.ash_instance();
    // Double-buffer: render into one while the consumer displays the other.
    let buffers = [
        DmabufTarget::new(&renderer.device, &instance, WIDTH, HEIGHT)?,
        DmabufTarget::new(&renderer.device, &instance, WIDTH, HEIGHT)?,
    ];
    let info = buffers[0].info;
    let fds = [buffers[0].raw_fd(), buffers[1].raw_fd()];

    let _ = std::fs::remove_file(SOCKET_PATH);
    let listener = UnixListener::bind(SOCKET_PATH).context("bind socket")?;
    eprintln!("[poc] dma-buf producer listening at {SOCKET_PATH}");

    let pointer = Arc::new(Mutex::new((WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0)));

    for conn in listener.incoming() {
        let stream = conn.context("accept")?;
        eprintln!("[poc] consumer connected");
        send_handshake(stream.as_raw_fd(), &info, &fds)?;

        {
            let mut reader = stream.try_clone().context("clone reader")?;
            let pointer = Arc::clone(&pointer);
            std::thread::spawn(move || {
                loop {
                    match Input::read(&mut reader) {
                        Ok(Input::PointerMove { x, y }) => *pointer.lock().unwrap() = (x, y),
                        Ok(other) => eprintln!("[poc] input: {other:?}"),
                        Err(_) => break,
                    }
                }
            });
        }

        let mut writer = stream.try_clone().context("clone writer")?;
        let mut index = 0usize;
        loop {
            let (px, py) = *pointer.lock().unwrap();
            renderer.render_into(px, py, &buffers[index].texture)?;
            if send_ready(&mut writer, index as u32).is_err() {
                eprintln!("[poc] consumer gone; waiting for next");
                break;
            }
            index ^= 1;
            std::thread::sleep(Duration::from_millis(16));
        }
    }
    Ok(())
}

// Headless consumer check: receive the dma-buf fds, drive the pointer, then
// mmap the shared buffer directly and confirm the triangle followed — proving
// fd passing + genuine cross-process memory sharing without needing GTK/GL.
fn run_dmabuf_probe() -> Result<()> {
    let stream = UnixStream::connect(SOCKET_PATH).context("connect (producer running?)")?;
    let (info, fds) = recv_handshake(stream.as_raw_fd())?;
    eprintln!(
        "[poc] handshake: {} fds, {}x{}, stride={}, modifier={:#x}, fourcc={:#x}",
        fds.len(),
        info.width,
        info.height,
        info.stride,
        info.modifier,
        info.fourcc
    );

    let target = (180.0f32, 70.0f32);
    Input::PointerMove {
        x: target.0,
        y: target.1,
    }
    .write(&mut (&stream))?;

    let mut reader = stream.try_clone().context("clone")?;
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut frames = 0u32;
    while Instant::now() < deadline {
        let mut tag = [0u8; 4];
        reader.read_exact(&mut tag).context("read ready tag")?;
        if &tag != b"RDY!" {
            bail!("unexpected notification tag {tag:?}");
        }
        let mut idx = [0u8; 4];
        reader.read_exact(&mut idx)?;
        let index = u32::from_le_bytes(idx) as usize;
        frames += 1;

        let pixel = read_dmabuf_pixel(fds[index].as_raw_fd(), &info, target.0 as u32, target.1 as u32)?;
        if pixel[0] > 180 && pixel[1] > 60 && pixel[1] < 160 && pixel[2] < 90 {
            eprintln!(
                "[poc] dmabuf-probe PASS: after {frames} frames, triangle at ({},{}) via \
                 mmap'd dma-buf — pixel ({},{},{},{})",
                target.0 as u32, target.1 as u32, pixel[0], pixel[1], pixel[2], pixel[3]
            );
            return Ok(());
        }
    }
    bail!("dmabuf-probe FAIL: triangle never reached target after {frames} frames");
}

/// mmap a dma-buf fd and read one RGBA pixel (CPU access; the exporter uses
/// host-visible memory in this POC).
fn read_dmabuf_pixel(fd: RawFd, info: &DmabufInfo, x: u32, y: u32) -> Result<[u8; 4]> {
    let len = (info.offset + info.stride * info.height) as usize;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        bail!("mmap dma-buf failed: {}", std::io::Error::last_os_error());
    }
    let offset = info.offset as usize + y as usize * info.stride as usize + x as usize * 4;
    let base = ptr as *const u8;
    let pixel = unsafe {
        [
            *base.add(offset),
            *base.add(offset + 1),
            *base.add(offset + 2),
            *base.add(offset + 3),
        ]
    };
    unsafe { libc::munmap(ptr, len) };
    Ok(pixel)
}

// Headless dma-buf export check: create a dma-buf-capable device, export an
// image, render into it, then map the exported memory and confirm the triangle
// landed — proving device creation + export + GPU write all work end to end.
async fn run_dmabuf_once() -> Result<()> {
    let renderer = Renderer::new(true).await?;
    let instance = renderer.ash_instance();
    let target = dmabuf_export::DmabufTarget::new(&renderer.device, &instance, WIDTH, HEIGHT)?;
    eprintln!(
        "[poc] dma-buf exported: fd={}, {}x{}, stride={}, offset={}, modifier={:#x}, fourcc={:#x}",
        target.raw_fd(),
        target.info.width,
        target.info.height,
        target.info.stride,
        target.info.offset,
        target.info.modifier,
        target.info.fourcc,
    );

    renderer.render_into(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0, &target.texture)?;

    let frame = target.map_and_read()?;
    let idx = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
    let px = &frame[idx..idx + 4];
    eprintln!(
        "[poc] dma-buf center pixel RGBA = ({}, {}, {}, {})",
        px[0], px[1], px[2], px[3]
    );
    // Triangle is orange (~229,102,26).
    if px[0] > 180 && px[1] > 60 && px[1] < 160 && px[2] < 90 {
        eprintln!("[poc] dmabuf-once PASS: triangle present in exported dma-buf");
        Ok(())
    } else {
        bail!("dmabuf-once FAIL: center pixel not orange (got {px:?})");
    }
}

async fn run_once() -> Result<()> {
    let r = Renderer::new(false).await?;
    let frame = r.render_readback(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0)?;
    let idx = ((HEIGHT / 2) * WIDTH + WIDTH / 2) as usize * 4;
    eprintln!(
        "[poc] center pixel RGBA = ({}, {}, {}, {})",
        frame[idx],
        frame[idx + 1],
        frame[idx + 2],
        frame[idx + 3]
    );
    eprintln!("[poc] once OK");
    Ok(())
}

async fn run_producer() -> Result<()> {
    let renderer = Renderer::new(false).await?;
    let _ = std::fs::remove_file(SOCKET_PATH);
    let listener = UnixListener::bind(SOCKET_PATH).context("bind socket")?;
    eprintln!("[poc] producer listening at {SOCKET_PATH}");

    // Triangle position, updated by the reader thread from incoming input.
    let pointer = Arc::new(Mutex::new((WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0)));

    for conn in listener.incoming() {
        let stream = conn.context("accept")?;
        eprintln!("[poc] consumer connected");
        let mut writer = stream.try_clone().context("clone stream")?;
        let mut reader = stream;

        // Input reader thread.
        {
            let pointer = Arc::clone(&pointer);
            std::thread::spawn(move || {
                loop {
                    match Input::read(&mut reader) {
                        Ok(Input::PointerMove { x, y }) => *pointer.lock().unwrap() = (x, y),
                        Ok(other) => eprintln!("[poc] input: {other:?}"),
                        Err(_) => break, // consumer disconnected
                    }
                }
            });
        }

        // Frame loop: ~60fps.
        loop {
            let (px, py) = *pointer.lock().unwrap();
            let frame = renderer.render_readback(px, py)?;
            if write_frame(&mut writer, WIDTH, HEIGHT, WIDTH * 4, &frame).is_err() {
                eprintln!("[poc] consumer gone; waiting for next");
                break;
            }
            std::thread::sleep(Duration::from_millis(16));
        }
    }
    Ok(())
}

// Headless end-to-end check: connect, move the pointer to a target, and verify
// the triangle's orange shows up there within a few frames.
fn run_probe() -> Result<()> {
    let mut stream = UnixStream::connect(SOCKET_PATH).context("connect (producer running?)")?;
    let mut input_stream = stream.try_clone().context("clone")?;

    let target = (180.0f32, 70.0f32);
    Input::PointerMove {
        x: target.0,
        y: target.1,
    }
    .write(&mut input_stream)?;

    let deadline = Instant::now() + Duration::from_secs(3);
    let mut frames = 0u32;
    while Instant::now() < deadline {
        let frame = read_frame(&mut stream)?;
        frames += 1;
        let x = target.0 as u32;
        let y = target.1 as u32;
        let idx = (y * frame.width + x) as usize * 4;
        let px = &frame.data[idx..idx + 4];
        // The triangle is orange (~229,102,26); background is ~(13,13,20).
        if px[0] > 180 && px[1] > 60 && px[1] < 160 && px[2] < 90 {
            eprintln!(
                "[poc] PASS: after {frames} frames, triangle followed pointer to \
                 ({},{}) — pixel ({},{},{},{})",
                x, y, px[0], px[1], px[2], px[3]
            );
            return Ok(());
        }
    }
    bail!("FAIL: triangle never reached target after {frames} frames");
}
