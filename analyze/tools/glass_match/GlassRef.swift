// Capture macOS's real Liquid Glass (NSGlassEffectView) over each backdrop PNG, plus the bare
// backdrop, at 2x through ScreenCaptureKit. The glass sits in a borderless non-activating
// floating panel: the state a HUD that never takes focus (like spotter_volume's card) is drawn
// in. It never takes focus, so the Mac stays usable while it runs.
//
// usage: GlassRef <backdrops dir> <out dir> <light|dark> [regular|clear]
// env:   ACTIVE=1  use a key window of the active app instead (dark glass is a bit lighter)
//
// Needs Screen Recording permission for the app that launches it (Terminal, Claude, ...).
import AppKit
import ScreenCaptureKit

let args = CommandLine.arguments
guard args.count >= 4 else {
    print("usage: GlassRef <backdrops dir> <out dir> <light|dark> [regular|clear]"); exit(2)
}
let inDir = URL(fileURLWithPath: args[1]), outDir = URL(fileURLWithPath: args[2])
let dark = args[3] == "dark"
let clear = args.count > 4 && args[4] == "clear"
let active = ProcessInfo.processInfo.environment["ACTIVE"] != nil
// must match gen_backdrops.py / render.py: card 296 x 128 logical, radius 22, 64 px margin
let CW: CGFloat = 296, CH: CGFloat = 128, M: CGFloat = 64, R: CGFloat = 22

class KeyWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

@MainActor
func run() async {
    let app = NSApplication.shared
    app.setActivationPolicy(active ? .regular : .accessory)
    let size = NSSize(width: CW + 2 * M, height: CH + 2 * M)
    let frame = NSRect(origin: NSPoint(x: 200, y: 300), size: size)
    let win: NSWindow = active
        ? KeyWindow(contentRect: frame, styleMask: [.borderless], backing: .buffered, defer: false)
        : NSPanel(contentRect: frame, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
    win.hasShadow = false
    win.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    let root = NSView(frame: NSRect(origin: .zero, size: size))
    let iv = NSImageView(frame: root.bounds)
    iv.imageScaling = .scaleAxesIndependently
    root.addSubview(iv)
    let glass = NSGlassEffectView(frame: NSRect(x: M, y: M, width: CW, height: CH))
    glass.cornerRadius = R
    if clear { glass.style = .clear }
    root.addSubview(glass)
    win.contentView = root
    if active {
        win.makeKeyAndOrderFront(nil)
        app.activate(ignoringOtherApps: true)
    } else {
        win.level = .floating
        win.ignoresMouseEvents = true
        win.orderFrontRegardless()
    }

    let files = try! FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "png" }.sorted { $0.path < $1.path }
    try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    try? await Task.sleep(nanoseconds: 500_000_000)

    let content = try! await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
    guard let scw = content.windows.first(where: { $0.windowID == CGWindowID(win.windowNumber) }) else {
        print("capture window not found (Screen Recording permission?)"); exit(1)
    }
    let filter = SCContentFilter(desktopIndependentWindow: scw)
    let cfg = SCStreamConfiguration()
    cfg.width = Int(size.width * 2); cfg.height = Int(size.height * 2)
    cfg.showsCursor = false
    cfg.ignoreShadowsSingleWindow = true
    cfg.colorSpaceName = CGColorSpace.sRGB

    // the glass has no temporal adaptation: it is final within a frame or two
    func grab(_ name: String) async {
        try? await Task.sleep(nanoseconds: 60_000_000)
        let img = try! await SCScreenshotManager.captureImage(contentFilter: filter, configuration: cfg)
        try! NSBitmapImageRep(cgImage: img).representation(using: .png, properties: [:])!
            .write(to: outDir.appendingPathComponent(name))
    }
    for f in files {
        let im = NSImage(contentsOf: f)!
        im.size = size                       // 2x pixels -> logical size
        iv.image = im
        let base = f.deletingPathExtension().lastPathComponent
        glass.isHidden = false
        await grab("\(base)_glass.png")
        glass.isHidden = true
        await grab("\(base)_bare.png")
    }
    print("captured \(files.count) backdrops (\(dark ? "dark" : "light")\(active ? ", key window" : "")) -> \(outDir.path)")
    exit(0)
}

Task { @MainActor in await run() }
NSApplication.shared.run()
