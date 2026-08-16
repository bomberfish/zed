use crate::{
    BoolExt, MacDispatcher, MacDisplay, MacKeyboardLayout, MacKeyboardMapper, MacWindow,
    embed::EmbedSocket,
    embed_window::{EmbedDisplay, EmbedWindow},
    events::key_to_native,
    ns_string,
    pasteboard::Pasteboard,
    renderer, set_active_window_cursor_style,
};
use anyhow::{Context as _, anyhow};
use block::ConcreteBlock;
use cocoa::{
    appkit::{
        NSApplication,
        NSApplicationActivationPolicy::{
            NSApplicationActivationPolicyAccessory, NSApplicationActivationPolicyRegular,
        },
        NSControl as _, NSEventModifierFlags, NSMenu, NSMenuItem, NSModalResponse, NSOpenPanel,
        NSSavePanel, NSVisualEffectState, NSVisualEffectView, NSWindow,
    },
    base::{BOOL, NO, YES, id, nil, selector},
    foundation::{
        NSArray, NSAutoreleasePool, NSBundle, NSInteger, NSProcessInfo, NSString, NSUInteger, NSURL,
    },
};
use core_foundation::{
    base::{CFRelease, CFType, CFTypeRef, OSStatus, TCFType},
    boolean::CFBoolean,
    data::CFData,
    dictionary::{CFDictionary, CFDictionaryRef, CFMutableDictionary},
    runloop::CFRunLoopRun,
    string::{CFString, CFStringRef},
};
use core_graphics::display::CGDirectDisplayID;
use ctor::ctor;
use dispatch2::DispatchQueue;
use futures::channel::oneshot;
use gpui::{
    Action, AnyWindowHandle, BackgroundExecutor, ClipboardItem, CursorStyle, ForegroundExecutor,
    KeyContext, Keymap, Menu, MenuItem, OsMenu, OwnedMenu, PathPromptOptions, Platform,
    PlatformDisplay, PlatformKeyboardLayout, PlatformKeyboardMapper, PlatformTextSystem,
    PlatformWindow, Result, SystemMenuType, Task, ThermalState, WindowAppearance, WindowKind,
    WindowParams, popup::PopupNotSupportedError,
};
use gpui_util::{ResultExt, new_std_command};
use itertools::Itertools;
use objc::{
    class,
    declare::ClassDecl,
    msg_send,
    runtime::{Class, Object, Sel},
    sel, sel_impl,
};
use parking_lot::Mutex;
use ptr::null_mut;
use semver::Version;
use std::{
    cell::Cell,
    ffi::{CStr, OsStr, c_void},
    os::{raw::c_char, unix::ffi::OsStrExt, unix::net::UnixStream},
    path::{Path, PathBuf},
    ptr,
    rc::Rc,
    slice, str,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[allow(non_upper_case_globals)]
const NSUTF8StringEncoding: NSUInteger = 4;

const MAC_PLATFORM_IVAR: &str = "platform";
static mut APP_CLASS: *const Class = ptr::null();
static mut APP_DELEGATE_CLASS: *const Class = ptr::null();

#[ctor(unsafe)]
unsafe fn build_classes() {
    unsafe {
        APP_CLASS = {
            let mut decl = ClassDecl::new("GPUIApplication", class!(NSApplication)).unwrap();
            decl.add_ivar::<*mut c_void>(MAC_PLATFORM_IVAR);
            decl.register()
        }
    };
    unsafe {
        APP_DELEGATE_CLASS = {
            let mut decl = ClassDecl::new("GPUIApplicationDelegate", class!(NSResponder)).unwrap();
            decl.add_ivar::<*mut c_void>(MAC_PLATFORM_IVAR);
            decl.add_method(
                sel!(applicationWillFinishLaunching:),
                will_finish_launching as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(applicationDidFinishLaunching:),
                did_finish_launching as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(applicationShouldHandleReopen:hasVisibleWindows:),
                should_handle_reopen as extern "C" fn(&mut Object, Sel, id, bool),
            );
            decl.add_method(
                sel!(applicationWillTerminate:),
                will_terminate as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(handleGPUIMenuItem:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            // Add menu item handlers so that OS save panels have the correct key commands
            decl.add_method(
                sel!(cut:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(copy:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(paste:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(selectAll:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(undo:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(redo:),
                handle_menu_item as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(validateMenuItem:),
                validate_menu_item as extern "C" fn(&mut Object, Sel, id) -> bool,
            );
            decl.add_method(
                sel!(menuWillOpen:),
                menu_will_open as extern "C" fn(&mut Object, Sel, id),
            );
            decl.add_method(
                sel!(applicationDockMenu:),
                handle_dock_menu as extern "C" fn(&mut Object, Sel, id) -> id,
            );
            decl.add_method(
                sel!(application:openURLs:),
                open_urls as extern "C" fn(&mut Object, Sel, id, id),
            );

            decl.add_method(
                sel!(onKeyboardLayoutChange:),
                on_keyboard_layout_change as extern "C" fn(&mut Object, Sel, id),
            );

            decl.add_method(
                sel!(onThermalStateChange:),
                on_thermal_state_change as extern "C" fn(&mut Object, Sel, id),
            );

            decl.add_method(
                sel!(onSystemWake:),
                on_system_wake as extern "C" fn(&mut Object, Sel, id),
            );

            decl.register()
        }
    }
}

pub struct MacPlatform(Mutex<MacPlatformState>);

pub(crate) struct MacPlatformState {
    background_executor: BackgroundExecutor,
    foreground_executor: ForegroundExecutor,
    text_system: Arc<dyn PlatformTextSystem>,
    renderer_context: renderer::Context,
    headless: bool,
    general_pasteboard: Pasteboard,
    find_pasteboard: Pasteboard,
    reopen: Option<Box<dyn FnMut()>>,
    on_keyboard_layout_change: Option<Box<dyn FnMut()>>,
    on_thermal_state_change: Option<Box<dyn FnMut()>>,
    on_system_wake: Option<Box<dyn FnMut()>>,
    system_wake_observer_registered: bool,
    quit: Option<Box<dyn FnMut()>>,
    menu_command: Option<Box<dyn FnMut(&dyn Action)>>,
    validate_menu_command: Option<Box<dyn FnMut(&dyn Action) -> bool>>,
    will_open_menu: Option<Box<dyn FnMut()>>,
    menu_actions: Vec<Box<dyn Action>>,
    open_urls: Option<Box<dyn FnMut(Vec<String>)>>,
    finish_launching: Option<Box<dyn FnOnce()>>,
    dock_menu: Option<id>,
    menus: Option<Vec<OwnedMenu>>,
    keyboard_mapper: Rc<MacKeyboardMapper>,
    /// Mirrors `[NSCursor setHiddenUntilMouseMoves:]` state, which AppKit doesn't expose.
    cursor_visible: Arc<AtomicBool>,
    system_notifications: crate::system_notifications::SystemNotificationState,
    /// Set when this process was started to serve an embed socket: windows bind to
    /// consumer connections instead of opening an `NSWindow`, and the cursor and
    /// pasteboard travel over those connections rather than to AppKit.
    embedding: bool,
    /// The consumer that owns the pasteboard bridge: the most recently bound one.
    ///
    /// One embed process serves every pane the host opens, and there is only one
    /// system clipboard to bridge to, so the newest pane wins — which is also the
    /// pane the user just interacted with. Linux has the same single global writer.
    embed_clipboard: Option<Arc<EmbedSocket>>,
}

impl MacPlatform {
    pub fn new(headless: bool) -> Self {
        let dispatcher = Arc::new(MacDispatcher::new());

        #[cfg(feature = "font-kit")]
        let text_system = Arc::new(crate::MacTextSystem::new());

        #[cfg(not(feature = "font-kit"))]
        let text_system = {
            if !headless {
                log::warn!(
                    "gpui_macos was compiled without the `font-kit` feature, so no text will be rendered."
                );
            }
            Arc::new(gpui::NoopTextSystem::new())
        };

        let keyboard_layout = MacKeyboardLayout::new();
        let keyboard_mapper = Rc::new(MacKeyboardMapper::new(keyboard_layout.id()));

        // The listener starts here rather than lazily in `open_window`, because the
        // order is the other way round: the host connects, the connection is
        // accepted, and *that* is what makes a window open.
        //
        // No `headless` override, unlike Windows. There, embed mode has to force
        // `headless` off because the flag means "no GPU, no DirectWrite" and an
        // embed needs both. Here `headless` reaches this platform only from
        // `gpui_platform::headless()`, which the embed launch path never calls, and
        // `ZED_HEADLESS` is read on Linux/FreeBSD alone.
        let embedding = match std::env::var(crate::embed::EMBED_ENDPOINT_VAR) {
            Ok(socket_path) if !socket_path.is_empty() => {
                crate::embed::ensure_embed_listener(&socket_path);
                true
            }
            _ => false,
        };

        Self(Mutex::new(MacPlatformState {
            headless,
            text_system,
            background_executor: BackgroundExecutor::new(dispatcher.clone()),
            foreground_executor: ForegroundExecutor::new(dispatcher),
            renderer_context: renderer::Context::default(),
            general_pasteboard: Pasteboard::general(),
            find_pasteboard: Pasteboard::find(),
            reopen: None,
            quit: None,
            menu_command: None,
            validate_menu_command: None,
            will_open_menu: None,
            menu_actions: Default::default(),
            open_urls: None,
            finish_launching: None,
            dock_menu: None,
            on_keyboard_layout_change: None,
            on_thermal_state_change: None,
            on_system_wake: None,
            system_wake_observer_registered: false,
            menus: None,
            keyboard_mapper,
            cursor_visible: Arc::new(AtomicBool::new(true)),
            system_notifications: crate::system_notifications::SystemNotificationState::new(),
            embedding,
            embed_clipboard: None,
        }))
    }

    /// Bind a queued consumer to an offscreen window and run the session: frames
    /// out on a fixed clock, forwarded input in, until either side goes away.
    fn open_embed_window(
        &self,
        options: WindowParams,
        socket: Arc<EmbedSocket>,
        reader: UnixStream,
    ) -> Box<dyn PlatformWindow> {
        let (foreground_executor, renderer_context) = {
            let mut guard = self.0.lock();
            guard.embed_clipboard = Some(socket.clone());
            (
                guard.foreground_executor.clone(),
                guard.renderer_context.clone(),
            )
        };

        let display = Rc::new(EmbedDisplay::new()) as Rc<dyn PlatformDisplay>;
        let window = EmbedWindow::new(options, display, renderer_context);

        // Attempted before the mode byte goes out, because the byte has to state
        // what actually happened: every step of the handoff (surface allocation,
        // the Metal wrapping, the bootstrap registration) can fail on a machine
        // where CPU frames still work, and the fallback has to be indistinguishable
        // from a consumer that never asked for zero copy.
        let zero_copy = std::env::var_os(crate::embed::EMBED_IOSURFACE_VAR).and_then(|_| {
            window
                .enable_zero_copy()
                .inspect_err(|error| {
                    log::error!(
                        "[zed-embed] the IOSurface transport is unavailable, \
                         falling back to CPU frames: {error:#}"
                    )
                })
                .ok()
        });

        // The mode byte comes before anything else, because that is the order the
        // consumer reads in. Mode 0 is CPU frames, which have no handshake to
        // follow: the surface-passing transports announce themselves here instead.
        if let Err(error) = crate::embed::send_mode(&socket, zero_copy.is_some()) {
            log::error!("[zed-embed] sending the transport mode failed: {error:#}");
        }

        match zero_copy {
            Some(handshake) => {
                if let Err(error) = crate::embed::send_iosurface_handshake(
                    &socket,
                    &handshake.service_name,
                    handshake.buffer_count,
                    handshake.max_width,
                    handshake.max_height,
                ) {
                    log::error!("[zed-embed] sending the IOSurface handshake failed: {error:#}");
                }
                window.set_on_ready(Arc::new({
                    let socket = socket.clone();
                    move |index, width, height| {
                        if let Err(error) = crate::embed::send_ready(&socket, index, width, height)
                        {
                            log::error!(
                                "[zed-embed] announcing a finished frame failed: {error:#}"
                            );
                        }
                    }
                }));
            }
            None => {
                window.set_on_frame(Box::new({
                    let socket = socket.clone();
                    move |pixels, width, height| {
                        if let Err(error) = crate::embed::send_frame(&socket, width, height, pixels)
                        {
                            log::error!("[zed-embed] sending a frame failed: {error:#}");
                        }
                    }
                }));
            }
        }
        window.set_on_title_change(Box::new({
            let socket = socket.clone();
            move |title| {
                if let Err(error) = crate::embed::send_title(&socket, title) {
                    log::error!("[zed-embed] title update failed: {error:#}");
                }
            }
        }));

        let receiver = crate::embed::spawn_reader(reader);
        let weak = window.downgrade();
        let (tick_tx, mut tick_rx) = futures::channel::mpsc::channel::<()>(0);
        spawn_embed_frame_clock(tick_tx);

        foreground_executor
            .spawn(async move {
                use futures::StreamExt as _;

                let mut input_state = crate::embed::EmbedInputState::default();
                let mut last_cursor: Option<&'static str> = None;
                loop {
                    if tick_rx.next().await.is_none() {
                        break;
                    }
                    let Some(window) = weak.upgrade() else {
                        break;
                    };
                    window.ensure_active();
                    loop {
                        match receiver.try_recv() {
                            Ok(input) => {
                                input_state.apply(input, &window);
                            }
                            Err(std::sync::mpsc::TryRecvError::Empty) => break,
                            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                log::info!("[zed-embed] consumer disconnected; closing");
                                socket.shutdown();
                                window.close();
                                return;
                            }
                        }
                    }
                    // Whatever this tick's layout asked for, at most one message.
                    let pending = crate::embed::PENDING_CURSOR.lock().take();
                    if let Some(name) = pending
                        && last_cursor != Some(name)
                    {
                        match crate::embed::send_cursor(&socket, name) {
                            Ok(()) => last_cursor = Some(name),
                            Err(error) => {
                                log::error!("[zed-embed] cursor update failed: {error:#}")
                            }
                        }
                    }
                    window.tick_frame();
                }
            })
            .detach();

        Box::new(window)
    }

    unsafe fn create_menu_bar(
        &self,
        menus: &Vec<Menu>,
        delegate: id,
        actions: &mut Vec<Box<dyn Action>>,
        keymap: &Keymap,
    ) -> id {
        unsafe {
            let application_menu = NSMenu::new(nil).autorelease();
            application_menu.setDelegate_(delegate);

            for menu_config in menus {
                let menu = NSMenu::new(nil).autorelease();
                let menu_title = ns_string(&menu_config.name);
                menu.setTitle_(menu_title);
                menu.setDelegate_(delegate);

                for item_config in &menu_config.items {
                    menu.addItem_(Self::create_menu_item(
                        item_config,
                        delegate,
                        actions,
                        keymap,
                    ));
                }

                let menu_item = NSMenuItem::new(nil).autorelease();
                menu_item.setTitle_(menu_title);
                menu_item.setSubmenu_(menu);
                application_menu.addItem_(menu_item);

                if menu_config.name == "Window" {
                    let app: id = msg_send![APP_CLASS, sharedApplication];
                    app.setWindowsMenu_(menu);
                }
            }

            application_menu
        }
    }

    unsafe fn create_dock_menu(
        &self,
        menu_items: Vec<MenuItem>,
        delegate: id,
        actions: &mut Vec<Box<dyn Action>>,
        keymap: &Keymap,
    ) -> id {
        unsafe {
            let dock_menu = NSMenu::new(nil);
            dock_menu.setDelegate_(delegate);
            for item_config in menu_items {
                dock_menu.addItem_(Self::create_menu_item(
                    &item_config,
                    delegate,
                    actions,
                    keymap,
                ));
            }

            dock_menu
        }
    }

    unsafe fn create_menu_item(
        item: &MenuItem,
        delegate: id,
        actions: &mut Vec<Box<dyn Action>>,
        keymap: &Keymap,
    ) -> id {
        static DEFAULT_CONTEXT: OnceLock<Vec<KeyContext>> = OnceLock::new();

        unsafe {
            match item {
                MenuItem::Separator => NSMenuItem::separatorItem(nil),
                MenuItem::Action {
                    name,
                    action,
                    os_action,
                    checked,
                    disabled,
                } => {
                    // Note that this is intentionally using earlier bindings, whereas typically
                    // later ones take display precedence. See the discussion on
                    // https://github.com/zed-industries/zed/issues/23621
                    let keystrokes = keymap
                        .bindings_for_action(action.as_ref())
                        .find_or_first(|binding| {
                            binding.predicate().is_none_or(|predicate| {
                                predicate.eval(DEFAULT_CONTEXT.get_or_init(|| {
                                    let mut workspace_context = KeyContext::new_with_defaults();
                                    workspace_context.add("Workspace");
                                    let mut pane_context = KeyContext::new_with_defaults();
                                    pane_context.add("Pane");
                                    let mut editor_context = KeyContext::new_with_defaults();
                                    editor_context.add("Editor");

                                    pane_context.extend(&editor_context);
                                    workspace_context.extend(&pane_context);
                                    vec![workspace_context]
                                }))
                            })
                        })
                        .map(|binding| binding.keystrokes());

                    let selector = match os_action {
                        Some(gpui::OsAction::Cut) => selector("cut:"),
                        Some(gpui::OsAction::Copy) => selector("copy:"),
                        Some(gpui::OsAction::Paste) => selector("paste:"),
                        Some(gpui::OsAction::SelectAll) => selector("selectAll:"),
                        // "undo:" and "redo:" are always disabled in our case, as
                        // we don't have a NSTextView/NSTextField to enable them on.
                        Some(gpui::OsAction::Undo) => selector("handleGPUIMenuItem:"),
                        Some(gpui::OsAction::Redo) => selector("handleGPUIMenuItem:"),
                        None => selector("handleGPUIMenuItem:"),
                    };

                    let item;
                    if let Some(keystrokes) = keystrokes {
                        if keystrokes.len() == 1 {
                            let keystroke = &keystrokes[0];
                            let mut mask = NSEventModifierFlags::empty();
                            for (modifier, flag) in &[
                                (
                                    keystroke.modifiers().platform,
                                    NSEventModifierFlags::NSCommandKeyMask,
                                ),
                                (
                                    keystroke.modifiers().control,
                                    NSEventModifierFlags::NSControlKeyMask,
                                ),
                                (
                                    keystroke.modifiers().alt,
                                    NSEventModifierFlags::NSAlternateKeyMask,
                                ),
                                (
                                    keystroke.modifiers().shift,
                                    NSEventModifierFlags::NSShiftKeyMask,
                                ),
                            ] {
                                if *modifier {
                                    mask |= *flag;
                                }
                            }

                            item = NSMenuItem::alloc(nil)
                                .initWithTitle_action_keyEquivalent_(
                                    ns_string(name),
                                    selector,
                                    ns_string(key_to_native(keystroke.key()).as_ref()),
                                )
                                .autorelease();
                            if Self::os_version() >= Version::new(12, 0, 0) {
                                let _: () = msg_send![item, setAllowsAutomaticKeyEquivalentLocalization: NO];
                            }
                            item.setKeyEquivalentModifierMask_(mask);
                        } else {
                            item = NSMenuItem::alloc(nil)
                                .initWithTitle_action_keyEquivalent_(
                                    ns_string(name),
                                    selector,
                                    ns_string(""),
                                )
                                .autorelease();
                        }
                    } else {
                        item = NSMenuItem::alloc(nil)
                            .initWithTitle_action_keyEquivalent_(
                                ns_string(name),
                                selector,
                                ns_string(""),
                            )
                            .autorelease();
                    }

                    if *checked {
                        item.setState_(NSVisualEffectState::Active);
                    }
                    item.setEnabled_(if *disabled { NO } else { YES });

                    let tag = actions.len() as NSInteger;
                    let _: () = msg_send![item, setTag: tag];
                    actions.push(action.boxed_clone());
                    item
                }
                MenuItem::Submenu(Menu {
                    name,
                    items,
                    disabled,
                }) => {
                    let item = NSMenuItem::new(nil).autorelease();
                    let submenu = NSMenu::new(nil).autorelease();
                    submenu.setDelegate_(delegate);
                    for item in items {
                        submenu.addItem_(Self::create_menu_item(item, delegate, actions, keymap));
                    }
                    item.setSubmenu_(submenu);
                    item.setEnabled_(if *disabled { NO } else { YES });
                    item.setTitle_(ns_string(name));
                    item
                }
                MenuItem::SystemMenu(OsMenu { name, menu_type }) => {
                    let item = NSMenuItem::new(nil).autorelease();
                    let submenu = NSMenu::new(nil).autorelease();
                    submenu.setDelegate_(delegate);
                    item.setSubmenu_(submenu);
                    item.setTitle_(ns_string(name));

                    match menu_type {
                        SystemMenuType::Services => {
                            let app: id = msg_send![APP_CLASS, sharedApplication];
                            app.setServicesMenu_(item);
                        }
                    }

                    item
                }
            }
        }
    }

    fn os_version() -> Version {
        let version = unsafe {
            let process_info = NSProcessInfo::processInfo(nil);
            process_info.operatingSystemVersion()
        };
        Version::new(
            version.majorVersion,
            version.minorVersion,
            version.patchVersion,
        )
    }
}

impl Platform for MacPlatform {
    fn background_executor(&self) -> BackgroundExecutor {
        self.0.lock().background_executor.clone()
    }

    fn foreground_executor(&self) -> gpui::ForegroundExecutor {
        self.0.lock().foreground_executor.clone()
    }

    fn text_system(&self) -> Arc<dyn PlatformTextSystem> {
        self.0.lock().text_system.clone()
    }

    fn run(&self, on_finish_launching: Box<dyn FnOnce()>) {
        let mut state = self.0.lock();
        if state.headless {
            drop(state);
            on_finish_launching();
            unsafe { CFRunLoopRun() };
        } else {
            state.finish_launching = Some(on_finish_launching);
            drop(state);
        }

        unsafe {
            let app: id = msg_send![APP_CLASS, sharedApplication];
            let app_delegate: id = msg_send![APP_DELEGATE_CLASS, new];
            app.setDelegate_(app_delegate);

            let self_ptr = self as *const Self as *const c_void;
            (*app).set_ivar(MAC_PLATFORM_IVAR, self_ptr);
            (*app_delegate).set_ivar(MAC_PLATFORM_IVAR, self_ptr);

            let pool = NSAutoreleasePool::new(nil);
            app.run();
            pool.drain();

            (*app).set_ivar(MAC_PLATFORM_IVAR, null_mut::<c_void>());
            (*NSWindow::delegate(app)).set_ivar(MAC_PLATFORM_IVAR, null_mut::<c_void>());
        }
    }

    fn quit(&self) {
        // Quitting the app causes us to close windows, which invokes `Window::on_close` callbacks
        // synchronously before this method terminates. If we call `Platform::quit` while holding a
        // borrow of the app state (which most of the time we will do), we will end up
        // double-borrowing the app state in the `on_close` callbacks for our open windows. To solve
        // this, we make quitting the application asynchronous so that we aren't holding borrows to
        // the app state on the stack when we actually terminate the app.

        unsafe {
            DispatchQueue::main().exec_async_f(ptr::null_mut(), quit);
        }

        extern "C" fn quit(_: *mut c_void) {
            unsafe {
                let app = NSApplication::sharedApplication(nil);
                let _: () = msg_send![app, terminate: nil];
            }
        }
    }

    fn restart(&self, binary_path: Option<PathBuf>) {
        use std::os::unix::process::CommandExt as _;

        let app_pid = std::process::id().to_string();
        let app_path = binary_path
            .or_else(|| {
                self.app_path()
                    .ok()
                    // When the app is not bundled, `app_path` returns the
                    // directory containing the executable. Disregard this
                    // and get the path to the executable itself.
                    .and_then(|path| (path.extension()?.to_str()? == "app").then_some(path))
            })
            .unwrap_or_else(|| std::env::current_exe().unwrap());

        // Wait until this process has exited and then re-open this path.
        let script = r#"
            while kill -0 $0 2> /dev/null; do
                sleep 0.1
            done
            open "$1"
        "#;

        #[allow(
            clippy::disallowed_methods,
            reason = "We are restarting ourselves, using std command thus is fine"
        )]
        let restart_process = new_std_command("/bin/bash")
            .arg("-c")
            .arg(script)
            .arg(app_pid)
            .arg(app_path)
            .process_group(0)
            .spawn();

        match restart_process {
            Ok(_) => self.quit(),
            Err(e) => log::error!("failed to spawn restart script: {:?}", e),
        }
    }

    fn activate(&self, ignoring_other_apps: bool) {
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            app.activateIgnoringOtherApps_(ignoring_other_apps.to_objc());
        }
    }

    fn hide(&self) {
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            let _: () = msg_send![app, hide: nil];
        }
    }

    fn hide_other_apps(&self) {
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            let _: () = msg_send![app, hideOtherApplications: nil];
        }
    }

    fn unhide_other_apps(&self) {
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            let _: () = msg_send![app, unhideAllApplications: nil];
        }
    }

    fn primary_display(&self) -> Option<Rc<dyn PlatformDisplay>> {
        Some(Rc::new(MacDisplay::primary()))
    }

    fn displays(&self) -> Vec<Rc<dyn PlatformDisplay>> {
        MacDisplay::all()
            .map(|screen| Rc::new(screen) as Rc<_>)
            .collect()
    }

    #[cfg(feature = "screen-capture")]
    fn is_screen_capture_supported(&self) -> bool {
        let min_version = cocoa::foundation::NSOperatingSystemVersion::new(12, 3, 0);
        crate::is_macos_version_at_least(min_version)
    }

    #[cfg(feature = "screen-capture")]
    fn screen_capture_sources(
        &self,
    ) -> oneshot::Receiver<Result<Vec<Rc<dyn gpui::ScreenCaptureSource>>>> {
        crate::screen_capture::get_sources()
    }

    fn active_window(&self) -> Option<AnyWindowHandle> {
        MacWindow::active_window()
    }

    // Returns the windows ordered front-to-back, meaning that the active
    // window is the first one in the returned vec.
    fn window_stack(&self) -> Option<Vec<AnyWindowHandle>> {
        Some(MacWindow::ordered_windows())
    }

    fn open_window(
        &self,
        handle: AnyWindowHandle,
        options: WindowParams,
    ) -> Result<Box<dyn PlatformWindow>> {
        // Native popups are not implemented on macOS yet. Rejecting lets callers fall back to
        // gpui's in-window popovers.
        if let WindowKind::AnchoredPopup(_) = options.kind {
            return Err(PopupNotSupportedError.into());
        }

        // An embed window replaces the native one outright: no `NSWindow`, and its
        // pixels go to whichever consumer connected over the embed socket. Windows
        // claims its single pipe with a flag; here each pane is its own connection
        // on one shared listener, as on Linux, so the only question is whether one
        // is waiting. Zed also opens windows of its own accord — settings being the
        // obvious one — and those still want a real window.
        let embedding = self.0.lock().embedding;
        if embedding {
            match crate::embed::take_pending_connection() {
                Some((socket, reader)) => {
                    return Ok(self.open_embed_window(options, Arc::new(socket), reader));
                }
                None => log::info!("[zed-embed] no consumer queued; opening a native window"),
            }
        }

        let (cursor_visible, foreground_executor, background_executor, renderer_context) = {
            let guard = self.0.lock();
            (
                guard.cursor_visible.clone(),
                guard.foreground_executor.clone(),
                guard.background_executor.clone(),
                guard.renderer_context.clone(),
            )
        };

        Ok(Box::new(MacWindow::open(
            handle,
            options,
            cursor_visible,
            foreground_executor,
            background_executor,
            renderer_context,
        )))
    }

    fn window_appearance(&self) -> WindowAppearance {
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            let appearance: id = msg_send![app, effectiveAppearance];
            crate::window_appearance::window_appearance_from_native(appearance)
        }
    }

    fn open_url(&self, url: &str) {
        unsafe {
            let ns_url = NSURL::alloc(nil).initWithString_(ns_string(url));
            if ns_url.is_null() {
                log::error!("Failed to create NSURL from string: {}", url);
                return;
            }
            let url = ns_url.autorelease();
            let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
            msg_send![workspace, openURL: url]
        }
    }

    fn register_url_scheme(&self, scheme: &str) -> Task<anyhow::Result<()>> {
        // API only available post Monterey
        // https://developer.apple.com/documentation/appkit/nsworkspace/3753004-setdefaultapplicationaturl
        let (done_tx, done_rx) = oneshot::channel();
        if Self::os_version() < Version::new(12, 0, 0) {
            return Task::ready(Err(anyhow!(
                "macOS 12.0 or later is required to register URL schemes"
            )));
        }

        let bundle_id = unsafe {
            let bundle: id = msg_send![class!(NSBundle), mainBundle];
            let bundle_id: id = msg_send![bundle, bundleIdentifier];
            if bundle_id == nil {
                return Task::ready(Err(anyhow!("Can only register URL scheme in bundled apps")));
            }
            bundle_id
        };

        unsafe {
            let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
            let scheme: id = ns_string(scheme);
            let app: id = msg_send![workspace, URLForApplicationWithBundleIdentifier: bundle_id];
            if app == nil {
                return Task::ready(Err(anyhow!(
                    "Cannot register URL scheme until app is installed"
                )));
            }
            let done_tx = Cell::new(Some(done_tx));
            let block = ConcreteBlock::new(move |error: id| {
                let result = if error == nil {
                    Ok(())
                } else {
                    let msg: id = msg_send![error, localizedDescription];
                    Err(anyhow!("Failed to register: {msg:?}"))
                };

                if let Some(done_tx) = done_tx.take() {
                    let _ = done_tx.send(result);
                }
            });
            let block = block.copy();
            let _: () = msg_send![workspace, setDefaultApplicationAtURL: app toOpenURLsWithScheme: scheme completionHandler: block];
        }

        self.background_executor()
            .spawn(async { done_rx.await.map_err(|e| anyhow!(e))? })
    }

    fn on_open_urls(&self, callback: Box<dyn FnMut(Vec<String>)>) {
        self.0.lock().open_urls = Some(callback);
    }

    fn prompt_for_paths(
        &self,
        options: PathPromptOptions,
    ) -> oneshot::Receiver<Result<Option<Vec<PathBuf>>>> {
        let (done_tx, done_rx) = oneshot::channel();
        self.foreground_executor()
            .spawn(async move {
                unsafe {
                    let panel = NSOpenPanel::openPanel(nil);
                    panel.setCanChooseDirectories_(options.directories.to_objc());
                    panel.setCanChooseFiles_(options.files.to_objc());
                    panel.setAllowsMultipleSelection_(options.multiple.to_objc());

                    panel.setCanCreateDirectories(true.to_objc());
                    panel.setResolvesAliases_(false.to_objc());
                    let done_tx = Cell::new(Some(done_tx));
                    let block = ConcreteBlock::new(move |response: NSModalResponse| {
                        let result = if response == NSModalResponse::NSModalResponseOk {
                            let mut result = Vec::new();
                            let urls = panel.URLs();
                            for i in 0..urls.count() {
                                let url = urls.objectAtIndex(i);
                                if url.isFileURL() == YES
                                    && let Ok(path) = ns_url_to_path(url)
                                {
                                    result.push(path)
                                }
                            }
                            Some(result)
                        } else {
                            None
                        };

                        if let Some(done_tx) = done_tx.take() {
                            let _ = done_tx.send(Ok(result));
                        }
                    });
                    let block = block.copy();

                    if let Some(prompt) = options.prompt {
                        let _: () = msg_send![panel, setPrompt: ns_string(&prompt)];
                    }

                    let _: () = msg_send![panel, beginWithCompletionHandler: block];
                }
            })
            .detach();
        done_rx
    }

    fn prompt_for_new_path(
        &self,
        directory: &Path,
        suggested_name: Option<&str>,
    ) -> oneshot::Receiver<Result<Option<PathBuf>>> {
        let directory = directory.to_owned();
        let suggested_name = suggested_name.map(|s| s.to_owned());
        let (done_tx, done_rx) = oneshot::channel();
        self.foreground_executor()
            .spawn(async move {
                unsafe {
                    let panel = NSSavePanel::savePanel(nil);
                    let path = ns_string(directory.to_string_lossy().as_ref());
                    let url = NSURL::fileURLWithPath_isDirectory_(nil, path, true.to_objc());
                    panel.setDirectoryURL(url);

                    if let Some(suggested_name) = suggested_name {
                        let name_string = ns_string(&suggested_name);
                        let _: () = msg_send![panel, setNameFieldStringValue: name_string];
                    }

                    let done_tx = Cell::new(Some(done_tx));
                    let block = ConcreteBlock::new(move |response: NSModalResponse| {
                        let mut result = None;
                        if response == NSModalResponse::NSModalResponseOk {
                            let url = panel.URL();
                            if url.isFileURL() == YES {
                                result = ns_url_to_path(panel.URL()).ok().map(|mut result| {
                                    let Some(filename) = result.file_name() else {
                                        return result;
                                    };
                                    let chunks = filename
                                        .as_bytes()
                                        .split(|&b| b == b'.')
                                        .collect::<Vec<_>>();

                                    // https://github.com/zed-industries/zed/issues/16969
                                    // Workaround a bug in macOS Sequoia that adds an extra file-extension
                                    // sometimes. e.g. `a.sql` becomes `a.sql.s` or `a.txtx` becomes `a.txtx.txt`
                                    //
                                    // This is conditional on OS version because I'd like to get rid of it, so that
                                    // you can manually create a file called `a.sql.s`. That said it seems better
                                    // to break that use-case than breaking `a.sql`.
                                    if chunks.len() == 3
                                        && chunks[1].starts_with(chunks[2])
                                        && Self::os_version() >= Version::new(15, 0, 0)
                                    {
                                        let new_filename = OsStr::from_bytes(
                                            &filename.as_bytes()
                                                [..chunks[0].len() + 1 + chunks[1].len()],
                                        )
                                        .to_owned();
                                        result.set_file_name(&new_filename);
                                    }
                                    result
                                })
                            }
                        }

                        if let Some(done_tx) = done_tx.take() {
                            let _ = done_tx.send(Ok(result));
                        }
                    });
                    let block = block.copy();
                    let _: () = msg_send![panel, beginWithCompletionHandler: block];
                }
            })
            .detach();

        done_rx
    }

    fn can_select_mixed_files_and_dirs(&self) -> bool {
        true
    }

    fn reveal_path(&self, path: &Path) {
        unsafe {
            let path = path.to_path_buf();
            self.0
                .lock()
                .background_executor
                .spawn(async move {
                    let full_path = ns_string(path.to_str().unwrap_or(""));
                    let root_full_path = ns_string("");
                    let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
                    let _: BOOL = msg_send![
                        workspace,
                        selectFile: full_path
                        inFileViewerRootedAtPath: root_full_path
                    ];
                })
                .detach();
        }
    }

    fn open_with_system(&self, path: &Path) {
        let path = path.to_owned();
        self.0
            .lock()
            .background_executor
            .spawn(async move {
                #[allow(
                    clippy::disallowed_methods,
                    reason = "running on a background thread, so blocking is fine"
                )]
                new_std_command("open")
                    .arg("--")
                    .arg(path)
                    .status()
                    .context("invoking open command")
                    .log_err();
            })
            .detach();
    }

    fn on_quit(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().quit = Some(callback);
    }

    fn on_reopen(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().reopen = Some(callback);
    }

    fn on_keyboard_layout_change(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().on_keyboard_layout_change = Some(callback);
    }

    fn on_app_menu_action(&self, callback: Box<dyn FnMut(&dyn Action)>) {
        self.0.lock().menu_command = Some(callback);
    }

    fn on_will_open_app_menu(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().will_open_menu = Some(callback);
    }

    fn on_validate_app_menu_command(&self, callback: Box<dyn FnMut(&dyn Action) -> bool>) {
        self.0.lock().validate_menu_command = Some(callback);
    }

    fn on_thermal_state_change(&self, callback: Box<dyn FnMut()>) {
        self.0.lock().on_thermal_state_change = Some(callback);
    }

    fn on_system_wake(&self, callback: Box<dyn FnMut()>) {
        let mut state = self.0.lock();
        state.on_system_wake = Some(callback);
        if state.system_wake_observer_registered {
            return;
        }
        drop(state);

        // SAFETY: APP_CLASS is registered during startup and returns the shared NSApplication.
        unsafe {
            let app: id = msg_send![APP_CLASS, sharedApplication];
            let delegate: id = msg_send![app, delegate];
            if delegate != nil {
                register_system_wake_observer(delegate);
                self.0.lock().system_wake_observer_registered = true;
            }
        }
    }

    fn thermal_state(&self) -> ThermalState {
        unsafe {
            let process_info: id = msg_send![class!(NSProcessInfo), processInfo];
            let state: NSInteger = msg_send![process_info, thermalState];
            match state {
                0 => ThermalState::Nominal,
                1 => ThermalState::Fair,
                2 => ThermalState::Serious,
                3 => ThermalState::Critical,
                _ => ThermalState::Nominal,
            }
        }
    }

    fn show_system_notification(&self, notification: gpui::SystemNotification) {
        let mut state = self.0.lock();
        let executor = state.foreground_executor.clone();
        state.system_notifications.show(&executor, notification);
    }

    fn dismiss_system_notification(&self, tag: &str) {
        let mut state = self.0.lock();
        let executor = state.foreground_executor.clone();
        state.system_notifications.dismiss(&executor, tag);
    }

    fn on_system_notification_response(
        &self,
        callback: Box<dyn FnMut(gpui::SystemNotificationResponse)>,
    ) {
        let mut state = self.0.lock();
        let executor = state.foreground_executor.clone();
        state.system_notifications.on_response(&executor, callback);
    }

    fn keyboard_layout(&self) -> Box<dyn PlatformKeyboardLayout> {
        Box::new(MacKeyboardLayout::new())
    }

    fn keyboard_mapper(&self) -> Rc<dyn PlatformKeyboardMapper> {
        self.0.lock().keyboard_mapper.clone()
    }

    fn app_path(&self) -> Result<PathBuf> {
        unsafe {
            let bundle: id = NSBundle::mainBundle();
            anyhow::ensure!(!bundle.is_null(), "app is not running inside a bundle");
            Ok(path_from_objc(msg_send![bundle, bundlePath]))
        }
    }

    fn set_menus(&self, menus: Vec<Menu>, keymap: &Keymap) {
        unsafe {
            let app: id = msg_send![APP_CLASS, sharedApplication];
            let mut state = self.0.lock();
            let actions = &mut state.menu_actions;
            let menu = self.create_menu_bar(&menus, NSWindow::delegate(app), actions, keymap);
            drop(state);
            app.setMainMenu_(menu);
        }
        self.0.lock().menus = Some(menus.into_iter().map(|menu| menu.owned()).collect());
    }

    fn get_menus(&self) -> Option<Vec<OwnedMenu>> {
        self.0.lock().menus.clone()
    }

    fn set_dock_menu(&self, menu: Vec<MenuItem>, keymap: &Keymap) {
        unsafe {
            let app: id = msg_send![APP_CLASS, sharedApplication];
            let mut state = self.0.lock();
            let actions = &mut state.menu_actions;
            let new = self.create_dock_menu(menu, NSWindow::delegate(app), actions, keymap);
            if let Some(old) = state.dock_menu.replace(new) {
                CFRelease(old as _)
            }
        }
    }

    fn add_recent_document(&self, path: &Path) {
        if let Some(path_str) = path.to_str() {
            unsafe {
                let document_controller: id =
                    msg_send![class!(NSDocumentController), sharedDocumentController];
                let url: id = NSURL::fileURLWithPath_(nil, ns_string(path_str));
                let _: () = msg_send![document_controller, noteNewRecentDocumentURL:url];
            }
        }
    }

    fn path_for_auxiliary_executable(&self, name: &str) -> Result<PathBuf> {
        unsafe {
            let bundle: id = NSBundle::mainBundle();
            anyhow::ensure!(!bundle.is_null(), "app is not running inside a bundle");
            let name = ns_string(name);
            let url: id = msg_send![bundle, URLForAuxiliaryExecutable: name];
            anyhow::ensure!(!url.is_null(), "resource not found");
            ns_url_to_path(url)
        }
    }

    /// Match cursor style to one of the styles available
    /// in macOS's [NSCursor](https://developer.apple.com/documentation/appkit/nscursor).
    fn set_cursor_style(&self, style: CursorStyle) {
        // There is no `NSWindow` to hit-test in an embed session, and the pointer the
        // user sees belongs to the host's window: record the name instead, and let
        // the frame clock forward at most one per tick. Setting `NSCursor` here would
        // change the cursor of a process that has nothing on screen.
        if self.0.lock().embedding {
            *crate::embed::PENDING_CURSOR.lock() = Some(crate::embed::cursor_name(style));
            return;
        }
        unsafe {
            set_active_window_cursor_style(style);
        }
    }

    fn hide_cursor_until_mouse_moves(&self) {
        let cursor_visible = self.0.lock().cursor_visible.clone();
        if !cursor_visible.swap(false, Ordering::Relaxed) {
            return;
        }
        unsafe {
            let _: () = msg_send![class!(NSCursor), setHiddenUntilMouseMoves: YES];
        }
    }

    fn is_cursor_visible(&self) -> bool {
        self.0.lock().cursor_visible.load(Ordering::Relaxed)
    }

    fn should_auto_hide_scrollbars(&self) -> bool {
        #[allow(non_upper_case_globals)]
        const NSScrollerStyleOverlay: NSInteger = 1;

        unsafe {
            let style: NSInteger = msg_send![class!(NSScroller), preferredScrollerStyle];
            style == NSScrollerStyleOverlay
        }
    }

    fn read_from_clipboard(&self) -> Option<ClipboardItem> {
        // Paste takes what the host last pushed, so it sees the clipboard of the
        // machine the user is typing on. Unlike the Linux and Windows producers this
        // process could reach a real pasteboard — it is a full GUI process — but the
        // pasteboard it would reach belongs to whichever machine the daemon spawned
        // it on, which for a remote session is not the user's.
        let state = self.0.lock();
        if state.embedding {
            return crate::embed::CLIPBOARD_CACHE
                .lock()
                .clone()
                .map(ClipboardItem::new_string);
        }
        state.general_pasteboard.read()
    }

    fn write_to_clipboard(&self, item: ClipboardItem) {
        let embed_clipboard = {
            let state = self.0.lock();
            if !state.embedding {
                state.general_pasteboard.write(item);
                return;
            }
            state.embed_clipboard.clone()
        };
        let Some(text) = item.text() else {
            return;
        };
        // Cached as well as forwarded, exactly as the other two producers do it.
        // Forwarding alone is not enough: the host records the text before putting it
        // on the system clipboard, precisely so the resulting change notification
        // does not echo back here — which means nothing else ever puts our OWN copy
        // in the cache, and `read_from_clipboard` above reads only the cache. Without
        // this, copying inside the embed and pasting inside it yields whatever the
        // host last pushed.
        *crate::embed::CLIPBOARD_CACHE.lock() = Some(text.clone());
        let Some(socket) = embed_clipboard else {
            return;
        };
        if let Err(error) = crate::embed::send_clipboard(&socket, &text) {
            log::error!("[zed-embed] forwarding a copy to the host failed: {error:#}");
        }
    }

    fn read_from_find_pasteboard(&self) -> Option<ClipboardItem> {
        let state = self.0.lock();
        state.find_pasteboard.read()
    }

    fn write_to_find_pasteboard(&self, item: ClipboardItem) {
        let state = self.0.lock();
        state.find_pasteboard.write(item);
    }

    fn write_credentials(&self, url: &str, username: &str, password: &[u8]) -> Task<Result<()>> {
        let url = url.to_string();
        let username = username.to_string();
        let password = password.to_vec();
        self.background_executor().spawn(async move {
            unsafe {
                use security::*;

                let url = CFString::from(url.as_str());
                let username = CFString::from(username.as_str());
                let password = CFData::from_buffer(&password);

                // First, check if there are already credentials for the given server. If so, then
                // update the username and password.
                let mut verb = "updating";
                let mut query_attrs = CFMutableDictionary::with_capacity(2);
                query_attrs.set(kSecClass as *const _, kSecClassInternetPassword as *const _);
                query_attrs.set(kSecAttrServer as *const _, url.as_CFTypeRef());

                let mut attrs = CFMutableDictionary::with_capacity(4);
                attrs.set(kSecClass as *const _, kSecClassInternetPassword as *const _);
                attrs.set(kSecAttrServer as *const _, url.as_CFTypeRef());
                attrs.set(kSecAttrAccount as *const _, username.as_CFTypeRef());
                attrs.set(kSecValueData as *const _, password.as_CFTypeRef());

                let mut status = SecItemUpdate(
                    query_attrs.as_concrete_TypeRef(),
                    attrs.as_concrete_TypeRef(),
                );

                // If there were no existing credentials for the given server, then create them.
                if status == errSecItemNotFound {
                    verb = "creating";
                    status = SecItemAdd(attrs.as_concrete_TypeRef(), ptr::null_mut());
                }
                anyhow::ensure!(status == errSecSuccess, "{verb} password failed: {status}");
            }
            Ok(())
        })
    }

    fn read_credentials(&self, url: &str) -> Task<Result<Option<(String, Vec<u8>)>>> {
        let url = url.to_string();
        self.background_executor().spawn(async move {
            let url = CFString::from(url.as_str());
            let cf_true = CFBoolean::true_value().as_CFTypeRef();

            unsafe {
                use security::*;

                // Find any credentials for the given server URL.
                let mut attrs = CFMutableDictionary::with_capacity(5);
                attrs.set(kSecClass as *const _, kSecClassInternetPassword as *const _);
                attrs.set(kSecAttrServer as *const _, url.as_CFTypeRef());
                attrs.set(kSecReturnAttributes as *const _, cf_true);
                attrs.set(kSecReturnData as *const _, cf_true);

                let mut result = CFTypeRef::from(ptr::null());
                let status = SecItemCopyMatching(attrs.as_concrete_TypeRef(), &mut result);
                match status {
                    security::errSecSuccess => {}
                    security::errSecItemNotFound | security::errSecUserCanceled => return Ok(None),
                    _ => anyhow::bail!("reading password failed: {status}"),
                }

                let result = CFType::wrap_under_create_rule(result)
                    .downcast::<CFDictionary>()
                    .context("keychain item was not a dictionary")?;
                let username = result
                    .find(kSecAttrAccount as *const _)
                    .context("account was missing from keychain item")?;
                let username = CFType::wrap_under_get_rule(*username)
                    .downcast::<CFString>()
                    .context("account was not a string")?;
                let password = result
                    .find(kSecValueData as *const _)
                    .context("password was missing from keychain item")?;
                let password = CFType::wrap_under_get_rule(*password)
                    .downcast::<CFData>()
                    .context("password was not a string")?;

                Ok(Some((username.to_string(), password.bytes().to_vec())))
            }
        })
    }

    fn delete_credentials(&self, url: &str) -> Task<Result<()>> {
        let url = url.to_string();

        self.background_executor().spawn(async move {
            unsafe {
                use security::*;

                let url = CFString::from(url.as_str());
                let mut query_attrs = CFMutableDictionary::with_capacity(2);
                query_attrs.set(kSecClass as *const _, kSecClassInternetPassword as *const _);
                query_attrs.set(kSecAttrServer as *const _, url.as_CFTypeRef());

                let status = SecItemDelete(query_attrs.as_concrete_TypeRef());
                anyhow::ensure!(status == errSecSuccess, "delete password failed: {status}");
            }
            Ok(())
        })
    }
}

/// What an embed session's frames are paced by, and what the tick carries.
///
/// The clock is deliberately not `background_executor.timer`. That timer runs on
/// the shared background pool — the same pool a language server saturates — so
/// with rust-analyzer working it fires however late the pool is behind, which
/// delays not just frames but the input drain, since both hang off one await.
/// That is what collapsed the embed to a few frames a second the moment the LSP
/// started, while the same binary in a native window stayed smooth.
struct EmbedFrameClock {
    ticks: futures::channel::mpsc::Sender<()>,
    /// The vsync subscription, kept so the clock can unsubscribe itself once the
    /// session is gone. `None` on a machine where the display link would not
    /// start, where [`spawn_embed_frame_clock`] falls back to a sleeping thread.
    frames: Option<crate::WindowFrameSource>,
    /// Which display the subscription above is on, so
    /// [`retarget_embed_frame_clock`] can tell a real move from the consumer
    /// re-stating where it already is.
    display: Option<CGDirectDisplayID>,
}

thread_local! {
    /// The live embed session's clock, for [`retarget_embed_frame_clock`].
    ///
    /// One per process because one process serves one embed session — `EmbedSession`
    /// is claimed by the first `open_window` and every later window opens normally —
    /// and thread-local because the pointer is only ever touched on the main thread:
    /// `spawn_embed_frame_clock` runs there from `open_window`, the tick handler is
    /// on the main queue, and the input drain that retargets is a
    /// `foreground_executor` task.
    static EMBED_FRAME_CLOCK: Cell<*mut EmbedFrameClock> =
        const { Cell::new(ptr::null_mut()) };
}

/// Move an embed session's frame pacing onto `display_id`, the display the consumer
/// says the pane is actually on.
///
/// The consumer is the only side that knows: a producer window is offscreen, so it
/// is on no `NSScreen` at all. Without this the clock stays on whatever
/// `CGMainDisplayID()` returned when the session started, which on a laptop driving
/// an external panel is the wrong display in the expensive direction — a 120Hz
/// built-in pacing a pane shown at 240Hz gives the user half the frames the panel
/// could draw, and with the lid shut it names a display that is switched off.
///
/// Cheap to call repeatedly: the consumer sends this on attach and on every screen
/// change, and a repeat of the current display returns without touching the
/// registry.
pub(crate) fn retarget_embed_frame_clock(display_id: u32) {
    let display_id = display_id as CGDirectDisplayID;
    let clock = EMBED_FRAME_CLOCK.get();
    if clock.is_null() {
        return;
    }
    // SAFETY: the pointer is the clock leaked by `spawn_embed_frame_clock`, which is
    // never freed, and this thread-local is only read on the thread that set it —
    // the same main thread the tick handler runs on, so there is no concurrent
    // borrow.
    let clock = unsafe { &mut *clock };
    if clock.display == Some(display_id) {
        return;
    }
    let Some(frames) = clock.frames.as_mut() else {
        // The display link never started, so frames are on the fallback timer and
        // there is no subscription to move. Said once per move rather than silently,
        // because it means the pacing the consumer just asked for is not happening.
        log::info!(
            "[zed-embed] consumer is on display {display_id} but frames are on the \
             fallback timer, so pacing cannot follow it"
        );
        return;
    };
    match frames.start(display_id) {
        Ok(()) => {
            clock.display = Some(display_id);
            log::info!(
                "[zed-embed] pacing frames on display {display_id} at {}",
                describe_refresh_rate(display_id)
            );
        }
        Err(error) => {
            // The old subscription is gone either way — `start` stops before it
            // subscribes — so this leaves the session with no vsync rather than the
            // previous display's. Recovering onto the display it came from is worth
            // the second attempt; if that fails too there is nothing left to try.
            log::error!("[zed-embed] could not pace on display {display_id}: {error:#}");
            if let Some(previous) = clock.display
                && frames.start(previous).is_err()
            {
                clock.display = None;
            }
        }
    }
}

/// A display's refresh rate as a log line: `"240Hz"`, or `"a variable rate"` for the
/// zero CoreGraphics reports for ProMotion and other adaptive-sync panels.
///
/// Only ever logged. It is here because the whole question that prompted the
/// consumer telling us its display — "it might be doing 120? i cant tell" — was
/// unanswerable from either side's logs.
fn describe_refresh_rate(display_id: CGDirectDisplayID) -> String {
    // SAFETY: `CGDisplayCopyDisplayMode` takes an id and returns either null or a
    // mode this owns, released below. Both calls are documented as safe on any
    // thread.
    unsafe {
        let mode = core_graphics::display::CGDisplayCopyDisplayMode(display_id);
        if mode.is_null() {
            return "an unknown rate".to_string();
        }
        let hertz = core_graphics::display::CGDisplayModeGetRefreshRate(mode);
        core_graphics::display::CGDisplayModeRelease(mode);
        if hertz <= 0.0 {
            "a variable rate".to_string()
        } else {
            format!("{hertz:.0}Hz")
        }
    }
}

impl EmbedFrameClock {
    /// Offer a tick, and report whether the session is still listening.
    ///
    /// `try_send` on a zero-capacity channel succeeds only when the receiver is
    /// already waiting. A full channel means the foreground is still busy with the
    /// previous frame, and dropping the tick rather than queueing it is the point:
    /// no backlog to burn through once it catches up.
    fn tick(&mut self) -> bool {
        match self.ticks.try_send(()) {
            Err(error) if error.is_disconnected() => false,
            _ => true,
        }
    }
}

/// Pace an embed session's frames on the vsync of the main display, the way a
/// native window's repaint is paced.
///
/// A fixed 16ms sleep used to stand in for this, which capped the embed at 60fps
/// and — worse on a variable-refresh panel — beat against the real vsync, so
/// trackpad scrolling on a 120Hz display juddered while the same editor in a
/// native window did not.
///
/// The main display only as an opening guess, because an offscreen window is on no
/// screen at all. It is a poor guess on a laptop driving an external panel — and
/// wrong in the expensive direction, not the harmless one, since a display slower
/// than the one the pane is on costs frames the consumer can never get back. The
/// consumer corrects it with a `Display` message as soon as it attaches; see
/// [`retarget_embed_frame_clock`].
fn spawn_embed_frame_clock(ticks: futures::channel::mpsc::Sender<()>) {
    let clock = Box::into_raw(Box::new(EmbedFrameClock {
        ticks,
        frames: None,
        display: None,
    }));
    EMBED_FRAME_CLOCK.set(clock);
    // SAFETY: `clock` was just allocated here and nothing else refers to it. The
    // dispatch source below targets the main queue, so `embed_frame_clock_tick`
    // runs serially on this thread and is the only other reader.
    let clock = unsafe { &mut *clock };
    let mut frames = crate::WindowFrameSource::new(
        clock as *mut EmbedFrameClock as *mut c_void,
        embed_frame_clock_tick,
    );
    let display_id = unsafe { core_graphics::display::CGMainDisplayID() };
    match frames.start(display_id) {
        Ok(()) => {
            log::info!(
                "[zed-embed] pacing frames on the main display {display_id} at {} until the \
                 consumer says which display the pane is on",
                describe_refresh_rate(display_id)
            );
            clock.frames = Some(frames);
            clock.display = Some(display_id);
        }
        Err(error) => {
            log::error!(
                "[zed-embed] the display link would not start, pacing frames on a \
                 timer instead: {error:#}"
            );
            // A plain sleeping thread cannot be starved by executor load either. It
            // paces at the slowest refresh worth calling smooth rather than at a
            // guess about this machine's panel, since nothing here can ask.
            let mut ticks = clock.ticks.clone();
            if let Err(error) = std::thread::Builder::new()
                .name("zed-embed-frame-clock".into())
                .spawn(move || {
                    loop {
                        if let Err(error) = ticks.try_send(())
                            && error.is_disconnected()
                        {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(16));
                    }
                })
            {
                log::error!("[zed-embed] spawning the frame clock failed: {error}");
            }
        }
    }
}

/// One vsync on the main display, delivered on the main queue.
extern "C" fn embed_frame_clock_tick(context: *mut c_void) {
    // SAFETY: `context` is the `EmbedFrameClock` leaked by
    // `spawn_embed_frame_clock`, which outlives every tick, and the dispatch
    // source carrying it runs its handler serially on the main queue.
    let clock = unsafe { &mut *(context as *mut EmbedFrameClock) };
    if clock.tick() {
        return;
    }
    // The session is over. Unsubscribing from inside the handler is safe — it only
    // takes the registry lock, which is what the main thread is supposed to do, and
    // the source itself stays alive because the clock is leaked — and it is worth
    // doing, since otherwise a closed embed would keep waking the main thread once
    // per vsync for as long as the process lives.
    if let Some(frames) = clock.frames.as_mut() {
        frames.stop();
    }
}

unsafe fn path_from_objc(path: id) -> PathBuf {
    let len = msg_send![path, lengthOfBytesUsingEncoding: NSUTF8StringEncoding];
    let bytes = unsafe { path.UTF8String() as *const u8 };
    let path = str::from_utf8(unsafe { slice::from_raw_parts(bytes, len) }).unwrap();
    PathBuf::from(path)
}

unsafe fn get_mac_platform(object: &mut Object) -> &MacPlatform {
    unsafe {
        let platform_ptr: *mut c_void = *object.get_ivar(MAC_PLATFORM_IVAR);
        assert!(!platform_ptr.is_null());
        &*(platform_ptr as *const MacPlatform)
    }
}

extern "C" fn will_finish_launching(_this: &mut Object, _: Sel, _: id) {
    unsafe {
        let user_defaults: id = msg_send![class!(NSUserDefaults), standardUserDefaults];

        // The autofill heuristic controller causes slowdown and high CPU usage.
        // We don't know exactly why. This disables the full heuristic controller.
        //
        // Adapted from: https://github.com/ghostty-org/ghostty/pull/8625
        let name = ns_string("NSAutoFillHeuristicControllerEnabled");
        let existing_value: id = msg_send![user_defaults, objectForKey: name];
        if existing_value == nil {
            let false_value: id = msg_send![class!(NSNumber), numberWithBool:false];
            let _: () = msg_send![user_defaults, setObject: false_value forKey: name];
        }
    }
}

extern "C" fn did_finish_launching(this: &mut Object, _: Sel, _: id) {
    unsafe {
        let app: id = msg_send![APP_CLASS, sharedApplication];
        // An embed session has nothing of its own on screen: Regular would put a
        // second Dock icon beside the host's and take focus off it at launch.
        // Accessory rather than Prohibited, because Zed still opens ordinary windows
        // in an embed session (settings, for one) and those have to be activatable.
        let activation_policy = if std::env::var_os(crate::embed::EMBED_ENDPOINT_VAR).is_some() {
            NSApplicationActivationPolicyAccessory
        } else {
            NSApplicationActivationPolicyRegular
        };
        app.setActivationPolicy_(activation_policy);

        let notification_center: *mut Object =
            msg_send![class!(NSNotificationCenter), defaultCenter];
        let name = ns_string("NSTextInputContextKeyboardSelectionDidChangeNotification");
        let _: () = msg_send![notification_center, addObserver: this as id
            selector: sel!(onKeyboardLayoutChange:)
            name: name
            object: nil
        ];

        let thermal_name = ns_string("NSProcessInfoThermalStateDidChangeNotification");
        let process_info: id = msg_send![class!(NSProcessInfo), processInfo];
        let _: () = msg_send![notification_center, addObserver: this as id
            selector: sel!(onThermalStateChange:)
            name: thermal_name
            object: process_info
        ];

        let observer = this as *mut Object as id;
        let platform = get_mac_platform(this);
        let callback = {
            let mut state = platform.0.lock();
            if state.on_system_wake.is_some() && !state.system_wake_observer_registered {
                register_system_wake_observer(observer);
                state.system_wake_observer_registered = true;
            }
            state.finish_launching.take()
        };
        if let Some(callback) = callback {
            callback();
        }
    }
}

unsafe fn register_system_wake_observer(observer: id) {
    // SAFETY: observer is an Objective-C object implementing onSystemWake:.
    unsafe {
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let workspace_center: *mut Object = msg_send![workspace, notificationCenter];
        let wake_name = ns_string("NSWorkspaceDidWakeNotification");
        let _: () = msg_send![workspace_center, addObserver: observer
            selector: sel!(onSystemWake:)
            name: wake_name
            object: nil
        ];
    }
}

extern "C" fn should_handle_reopen(this: &mut Object, _: Sel, _: id, has_open_windows: bool) {
    if !has_open_windows {
        let platform = unsafe { get_mac_platform(this) };
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.reopen.take() {
            drop(lock);
            callback();
            platform.0.lock().reopen.get_or_insert(callback);
        }
    }
}

extern "C" fn will_terminate(this: &mut Object, _: Sel, _: id) {
    let platform = unsafe { get_mac_platform(this) };
    let mut lock = platform.0.lock();
    if let Some(mut callback) = lock.quit.take() {
        drop(lock);
        callback();
        platform.0.lock().quit.get_or_insert(callback);
    }
}

extern "C" fn on_keyboard_layout_change(this: &mut Object, _: Sel, _: id) {
    let platform = unsafe { get_mac_platform(this) };
    let mut lock = platform.0.lock();
    let keyboard_layout = MacKeyboardLayout::new();
    lock.keyboard_mapper = Rc::new(MacKeyboardMapper::new(keyboard_layout.id()));
    if let Some(mut callback) = lock.on_keyboard_layout_change.take() {
        drop(lock);
        callback();
        platform
            .0
            .lock()
            .on_keyboard_layout_change
            .get_or_insert(callback);
    }
}

extern "C" fn on_thermal_state_change(this: &mut Object, _: Sel, _: id) {
    // Defer to the next run loop iteration to avoid re-entrant borrows of the App RefCell,
    // as NSNotificationCenter delivers this notification synchronously and it may fire while
    // the App is already borrowed (same pattern as quit() above).
    let platform = unsafe { get_mac_platform(this) };
    let platform_ptr = platform as *const MacPlatform as *mut c_void;
    unsafe {
        DispatchQueue::main().exec_async_f(platform_ptr, on_thermal_state_change);
    }

    extern "C" fn on_thermal_state_change(context: *mut c_void) {
        let platform = unsafe { &*(context as *const MacPlatform) };
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.on_thermal_state_change.take() {
            drop(lock);
            callback();
            platform
                .0
                .lock()
                .on_thermal_state_change
                .get_or_insert(callback);
        }
    }
}

extern "C" fn on_system_wake(this: &mut Object, _: Sel, _: id) {
    // SAFETY: this is the registered app delegate carrying MAC_PLATFORM_IVAR.
    let platform = unsafe { get_mac_platform(this) };
    let platform_ptr = platform as *const MacPlatform as *mut c_void;
    // SAFETY: platform lives for the process lifetime while callbacks are registered.
    unsafe {
        DispatchQueue::main().exec_async_f(platform_ptr, on_system_wake);
    }

    extern "C" fn on_system_wake(context: *mut c_void) {
        // SAFETY: context is the MacPlatform pointer queued above.
        let platform = unsafe { &*(context as *const MacPlatform) };
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.on_system_wake.take() {
            drop(lock);
            callback();
            platform.0.lock().on_system_wake.get_or_insert(callback);
        }
    }
}

extern "C" fn open_urls(this: &mut Object, _: Sel, _: id, urls: id) {
    let urls = unsafe {
        (0..urls.count())
            .filter_map(|i| {
                let url = urls.objectAtIndex(i);
                match CStr::from_ptr(url.absoluteString().UTF8String() as *mut c_char).to_str() {
                    Ok(string) => Some(string.to_string()),
                    Err(err) => {
                        log::error!("error converting path to string: {}", err);
                        None
                    }
                }
            })
            .collect::<Vec<_>>()
    };
    let platform = unsafe { get_mac_platform(this) };
    let mut lock = platform.0.lock();
    if let Some(mut callback) = lock.open_urls.take() {
        drop(lock);
        callback(urls);
        platform.0.lock().open_urls.get_or_insert(callback);
    }
}

extern "C" fn handle_menu_item(this: &mut Object, _: Sel, item: id) {
    unsafe {
        let platform = get_mac_platform(this);
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.menu_command.take() {
            let tag: NSInteger = msg_send![item, tag];
            let index = tag as usize;
            if let Some(action) = lock.menu_actions.get(index) {
                let action = action.boxed_clone();
                drop(lock);
                callback(&*action);
            }
            platform.0.lock().menu_command.get_or_insert(callback);
        }
    }
}

extern "C" fn validate_menu_item(this: &mut Object, _: Sel, item: id) -> bool {
    unsafe {
        let mut result = false;
        let platform = get_mac_platform(this);
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.validate_menu_command.take() {
            let tag: NSInteger = msg_send![item, tag];
            let index = tag as usize;
            if let Some(action) = lock.menu_actions.get(index) {
                let action = action.boxed_clone();
                drop(lock);
                result = callback(action.as_ref());
            }
            platform
                .0
                .lock()
                .validate_menu_command
                .get_or_insert(callback);
        }
        result
    }
}

extern "C" fn menu_will_open(this: &mut Object, _: Sel, _: id) {
    unsafe {
        let platform = get_mac_platform(this);
        let mut lock = platform.0.lock();
        if let Some(mut callback) = lock.will_open_menu.take() {
            drop(lock);
            callback();
            platform.0.lock().will_open_menu.get_or_insert(callback);
        }
    }
}

extern "C" fn handle_dock_menu(this: &mut Object, _: Sel, _: id) -> id {
    unsafe {
        let platform = get_mac_platform(this);
        let state = platform.0.lock();
        if let Some(id) = state.dock_menu {
            id
        } else {
            nil
        }
    }
}

unsafe fn ns_url_to_path(url: id) -> Result<PathBuf> {
    let path: *mut c_char = msg_send![url, fileSystemRepresentation];
    anyhow::ensure!(!path.is_null(), "url is not a file path: {}", unsafe {
        CStr::from_ptr(url.absoluteString().UTF8String()).to_string_lossy()
    });
    Ok(PathBuf::from(OsStr::from_bytes(unsafe {
        CStr::from_ptr(path).to_bytes()
    })))
}

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    pub(super) fn TISCopyCurrentKeyboardLayoutInputSource() -> *mut Object;
    pub(super) fn TISCopyCurrentKeyboardInputSource() -> *mut Object;
    pub(super) fn TISGetInputSourceProperty(
        inputSource: *mut Object,
        propertyKey: *const c_void,
    ) -> *mut Object;

    pub(super) fn UCKeyTranslate(
        keyLayoutPtr: *const ::std::os::raw::c_void,
        virtualKeyCode: u16,
        keyAction: u16,
        modifierKeyState: u32,
        keyboardType: u32,
        keyTranslateOptions: u32,
        deadKeyState: *mut u32,
        maxStringLength: usize,
        actualStringLength: *mut usize,
        unicodeString: *mut u16,
    ) -> u32;
    pub(super) fn LMGetKbdType() -> u16;
    pub(super) static kTISPropertyUnicodeKeyLayoutData: CFStringRef;
    pub(super) static kTISPropertyInputSourceID: CFStringRef;
    pub(super) static kTISPropertyLocalizedName: CFStringRef;
    pub(super) static kTISPropertyInputSourceIsASCIICapable: CFStringRef;
    pub(super) static kTISPropertyInputSourceType: CFStringRef;
    pub(super) static kTISTypeKeyboardInputMode: CFStringRef;
}

mod security {
    #![allow(non_upper_case_globals)]
    use super::*;

    #[link(name = "Security", kind = "framework")]
    unsafe extern "C" {
        pub static kSecClass: CFStringRef;
        pub static kSecClassInternetPassword: CFStringRef;
        pub static kSecAttrServer: CFStringRef;
        pub static kSecAttrAccount: CFStringRef;
        pub static kSecValueData: CFStringRef;
        pub static kSecReturnAttributes: CFStringRef;
        pub static kSecReturnData: CFStringRef;

        pub fn SecItemAdd(attributes: CFDictionaryRef, result: *mut CFTypeRef) -> OSStatus;
        pub fn SecItemUpdate(query: CFDictionaryRef, attributes: CFDictionaryRef) -> OSStatus;
        pub fn SecItemDelete(query: CFDictionaryRef) -> OSStatus;
        pub fn SecItemCopyMatching(query: CFDictionaryRef, result: *mut CFTypeRef) -> OSStatus;
    }

    pub const errSecSuccess: OSStatus = 0;
    pub const errSecUserCanceled: OSStatus = -128;
    pub const errSecItemNotFound: OSStatus = -25300;
}
