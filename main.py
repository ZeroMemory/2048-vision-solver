import cv2
import numpy as np
import pyttsx3
import time
import argparse
import ctypes
import json
import hashlib
import shutil
import os
import stat
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from http.client import HTTPException
import pytesseract
from PIL import Image

VERSION = "1.0.0"
DIRECTIONS = {0: "left", 1: "up", 2: "right", 3: "down"}

def enable_dpi_awareness():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)

# ----------------------------
# Capture frame from IP webcam
# ----------------------------

class FrameUnavailable(ValueError):
    pass


def capture_from_ipcam(url):
    try:
        with urlopen(url, timeout=5) as response:
            content = response.read(10 * 1024 * 1024 + 1)
    except HTTPError as error:
        if error.code not in (408, 429) and error.code < 500:
            raise
        raise FrameUnavailable(str(error)) from error
    except URLError as error:
        if (urlsplit(url).scheme.lower() not in ("http", "https")
                or isinstance(error.reason, (str, ssl.SSLCertVerificationError))):
            raise
        raise FrameUnavailable(str(error)) from error
    except (OSError, HTTPException) as error:
        if urlsplit(url).scheme.lower() not in ("http", "https"):
            raise
        raise FrameUnavailable(str(error)) from error
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Camera image exceeds 10 MiB")
    try:
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) if content else None
    except cv2.error as error:
        raise FrameUnavailable("Camera response is not a decodable image") from error
    if frame is None:
        raise FrameUnavailable("Camera response is not a decodable image")
    return frame

# ----------------------------
# Adaptive board detection
# ----------------------------

def order_points(pts):
    points = np.asarray(pts, dtype=np.float32)
    if points.shape != (4,2) or not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4:
        raise ValueError("Four distinct finite board corners are required")
    hull = cv2.convexHull(points)
    if len(hull) != 4 or cv2.contourArea(hull) < 1:
        raise ValueError("Board corners must form a convex quadrilateral with nonzero area")
    center = points.mean(axis=0)
    angles = np.arctan2(points[:,1]-center[1], points[:,0]-center[0])
    ordered = points[np.argsort(angles)]
    start = min(range(4), key=lambda i: (float(ordered[i].sum()), float(ordered[i,1]), float(ordered[i,0])))
    return np.roll(ordered, -start, axis=0)

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0]-bl[0], br[1]-bl[1])
    widthB = np.hypot(tr[0]-tl[0], tr[1]-tl[1])
    maxWidth = int(max(widthA, widthB))
    heightA = np.hypot(tr[0]-br[0], tr[1]-br[1])
    heightB = np.hypot(tl[0]-bl[0], tl[1]-bl[1])
    maxHeight = int(max(heightA, heightB))
    if min(maxWidth, maxHeight) < 2:
        raise ValueError("Board dimensions must be at least two pixels")
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_and_warp(frame, blur_ksize=(5,5), canny_thresholds=(50,150), approx_eps_factor=0.02, min_contour_area=10000):
    vis = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_ksize, 0)
    edges = cv2.Canny(gray, canny_thresholds[0], canny_thresholds[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, vis

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    board_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps_factor * peri, True)
        if len(approx) == 4:
            board_cnt = approx.reshape(4,2).astype(np.float32)
            break

    if board_cnt is None:
        union = np.vstack([c for c in contours[:5]]) if len(contours) >= 1 else None
        if union is not None:
            hull = cv2.convexHull(union)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, approx_eps_factor * peri, True)
            if len(approx) == 4:
                board_cnt = approx.reshape(4,2).astype(np.float32)

    if board_cnt is None:
        return None, vis

    overlay = vis.copy()
    cv2.drawContours(overlay, [board_cnt.astype(int)], -1, (0,255,0), -1)
    alpha = 0.25
    cv2.addWeighted(overlay, alpha, vis, 1-alpha, 0, vis)
    cv2.polylines(vis, [board_cnt.astype(int)], True, (0,200,0), 3)

    warped = four_point_transform(frame, board_cnt)
    return warped, vis

# ----------------------------
# Split to cells, recognize values
# ----------------------------

def split_cells(board_img):
    h, w, _ = board_img.shape
    cell_h = h // 4
    cell_w = w // 4
    cells = []
    for i in range(4):
        for j in range(4):
            cell = board_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cells.append(cell)
    return cells

def recognize_cell_value(cell):
    # Ignore the grid border. Smooth camera/display moire before segmentation.
    h, w = cell.shape[:2]
    if min(h, w) < 20:
        raise ValueError("Board cells are too small; use a sharper/larger image")
    crop = cell[int(h*.13):int(h*.87), int(w*.13):int(w*.87)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = np.percentile(gray, [2, 98])
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Digits occupy the minority, regardless of light/dark theme.
    if np.mean(binary == 255) > .5:
        binary = 255 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)
    content = False
    ambiguous = False
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if x <= 0 or y <= 0 or x+width >= crop.shape[1] or y+height >= crop.shape[0]:
            continue
        if area < crop.shape[0]*crop.shape[1]*.001 or height < crop.shape[0]*.04:
            continue
        component = np.uint8(labels == index)
        ring = (cv2.dilate(component, np.ones((15,15), np.uint8))-component).astype(bool)
        contrast = abs(float(gray[component.astype(bool)].mean())-float(gray[ring].mean()))
        # Local contrast distinguishes a glyph from gradual illumination or display moire.
        if contrast < 7:
            continue
        if (high-low < 25 or height <= crop.shape[0]*.20 or area <= crop.shape[0]*crop.shape[1]*.008
                or width >= crop.shape[1]*.90 or height >= crop.shape[0]*.95):
            ambiguous = True
        else:
            mask[labels == index] = 255
            content = True
    if ambiguous:
        raise ValueError("Faint or undersized occupied tile; use a clearer board image")
    if not content:
        return 0
    # OCR the segmented foreground with ample white padding.
    prepared = cv2.resize(255-mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    prepared = cv2.copyMakeBorder(prepared, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    for mode in (7, 8, 10):
        text = pytesseract.image_to_string(Image.fromarray(prepared), lang="eng",
            config=f"--psm {mode} -c tessedit_char_whitelist=0123456789", timeout=5).strip()
        if text.isdigit():
            value = int(text)
            if 2 <= value <= 131072 and not value & (value-1):
                return value
    raise ValueError("Unreadable or invalid occupied tile; no move will be recommended")


def build_state(cells):
    values = [recognize_cell_value(c) for c in cells]
    grid = np.array(values).reshape(4,4)
    return grid

# ----------------------------
# Expectimax search
# ----------------------------

def get_moves(): return [0,1,2,3]

def move(grid, direction):
    if direction not in DIRECTIONS:
        raise ValueError("Direction must be 0:left, 1:up, 2:right, 3:down")
    g = np.rot90(np.array(grid, dtype=np.int64), direction).copy()
    for i in range(4):
        values = [int(value) for value in g[i] if value]
        merged = []
        j = 0
        while j < len(values):
            if j+1 < len(values) and values[j] == values[j+1]:
                merged.append(values[j]*2)
                j += 2
            else:
                merged.append(values[j])
                j += 1
        g[i] = merged + [0] * (4-len(merged))
    return np.rot90(g, -direction).copy()

def is_move_possible(grid,direction): return not np.array_equal(grid,move(grid,direction))

def heuristic(grid):
    empty = np.sum(grid==0)
    mono = sum([abs(row[i]-row[i+1]) for row in grid for i in range(3)])
    return empty*100 - mono

def expectimax(grid,depth,is_player):
    if depth==0: return float(heuristic(grid))
    if is_player:
        best=-1e9
        for d in get_moves():
            if is_move_possible(grid,d): best=max(best,expectimax(move(grid,d),depth-1,False))
        return float(heuristic(grid)) if best == -1e9 else best
    else:
        empty=list(zip(*np.where(grid==0)))
        if not empty: return heuristic(grid)
        exp=0
        for (i,j) in empty:
            for tile,p in [(2,0.9),(4,0.1)]:
                g2=np.copy(grid); g2[i,j]=tile
                exp += p*expectimax(g2,depth-1,True)/len(empty)
        return exp

def best_move(grid):
    best=-float("inf"); best_dir=None
    for d in get_moves():
        if is_move_possible(grid,d):
            val=expectimax(move(grid,d),3,False)
            if val>best: best=val; best_dir=d
    return best_dir

# ----------------------------
# TTS using pyttsx3
# ----------------------------

def speak(text):
    # SAPI is called on its owning thread; speech never races a second utterance.
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ----------------------------
# Visualization
# ----------------------------

def visualize_grid(board_img, grid, direction):
    vis = board_img.copy()
    h, w, _ = vis.shape
    cell_h = h//4
    cell_w = w//4
    for i in range(4):
        for j in range(4):
            y0=i*cell_h; y1=(i+1)*cell_h
            x0=j*cell_w; x1=(j+1)*cell_w
            cv2.rectangle(vis, (x0,y0),(x1,y1),(255,0,0),2)
            val = grid[i,j]
            if val>0: cv2.putText(vis,str(val),(x0+10,y0+cell_h//2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    label = DIRECTIONS[direction] if direction is not None else "game over"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow("2048 Vision Solver", vis)


def open_windows():
    for name, position in [("2048 Board Detection", (30, 30)), ("2048 Vision Solver", (700, 30))]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600, 600)
        cv2.moveWindow(name, *position)

# ----------------------------
# Main loop
# ----------------------------

def analyze(frame, board_crop=False):
    if board_crop:
        warped, annotated = frame, frame.copy()
    else:
        warped, annotated = detect_and_warp(frame)
        if warped is None:
            raise FrameUnavailable("No four-corner board found; use --board-crop for an already cropped board")
    state = build_state(split_cells(warped))
    direction = best_move(state)
    return warped, annotated, state, direction


def ensure_runtime_output(path):
    output = Path(path)
    sources = [__file__, sys.executable] + [vars(module).get('__file__') for module in tuple(sys.modules.values()) if module is not None]
    for filename in sources:
        if filename:
            source = Path(filename)
            if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
                raise ValueError('Output must not replace runtime source files')


def output_identity(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_junction() or not stat.S_ISREG(info.st_mode):
        raise ValueError("Output must be a regular file without links")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, digest.digest()


def save_result(path, encoded, previous):
    workspace = Path(tempfile.mkdtemp(prefix=".2048-result-", dir=path.parent))
    temporary, backup = workspace / "result", workspace / "previous"
    moved = preserve = False
    try:
        temporary.write_text(encoded + "\n", encoding="utf-8")
        identity = output_identity(temporary)
        if output_identity(path) != previous:
            raise ValueError("Output changed during capture; refusing to replace it")
        if previous is not None:
            path.rename(backup)
            moved = True
            if output_identity(backup) != previous:
                raise ValueError("Output changed during capture; refusing to replace it")
        if os.name == 'nt':
            temporary.rename(path)
        else:
            os.link(temporary, path)
        return identity
    except BaseException as error:
        if moved:
            try:
                if os.name == 'nt':
                    backup.rename(path)
                else:
                    os.link(backup, path)
            except OSError as restore_error:
                preserve = True
                raise OSError(f"Could not restore output; previous file: {backup}") from restore_error
        raise
    finally:
        if not preserve:
            shutil.rmtree(workspace)


def main():
    enable_dpi_awareness()
    parser = argparse.ArgumentParser(description="Read a 2048 board and recommend an expectimax move")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera-url", help="Your IP webcam snapshot URL")
    parser.add_argument("--board-crop", action="store_true", help="Input is already cropped to the board")
    parser.add_argument("--tesseract-cmd", help="Path to tesseract.exe if absent from PATH")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--overwrite-output", action="store_true", help="Replace an existing output JSON file")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--once", action="store_true", help="Read one camera snapshot, then exit")
    parser.add_argument("--speak", action="store_true", help="Speak the recommendation using Windows SAPI")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    if args.output_json and (args.output_json.is_reserved() or ':' in args.output_json.name or args.output_json.name.rstrip(' .') != args.output_json.name):
        parser.error("Output must use a regular filename")
    try:
        if args.output_json:
            ensure_runtime_output(args.output_json)
    except ValueError as error:
        parser.error(str(error))
    if args.image and args.output_json and (args.output_json.resolve() == args.image.resolve() or (
        args.output_json.exists() and args.image.exists() and args.output_json.samefile(args.image)
    )):
        parser.error("Output must not replace the input image")
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    command = pytesseract.pytesseract.tesseract_cmd
    executable = Path(shutil.which(command) or command)
    data_roots = [executable.parent / 'tessdata']
    if os.environ.get('TESSDATA_PREFIX'):
        data_roots.append(Path(os.environ['TESSDATA_PREFIX']))
    protected = [executable, *(path for root in data_roots for path in root.glob('*.traineddata'))]
    if args.output_json and any(args.output_json.resolve() == path.resolve() or (
        args.output_json.exists() and path.exists() and args.output_json.samefile(path)
    ) for path in protected):
        parser.error("Output must not replace the Tesseract executable or language data")
    try:
        previous_output = output_identity(args.output_json) if args.output_json else None
        if previous_output is not None and not args.overwrite_output:
            raise ValueError("Output already exists; choose a new file or pass --overwrite-output")
        if "eng" not in pytesseract.get_languages():
            raise ValueError("Tesseract English (eng) language data is required")
        gui_initialized = False
        if not args.no_gui and not args.image and not args.once:
            open_windows()
            gui_initialized = True
        while True:
            try:
                if args.image:
                    content = args.image.read_bytes()
                    frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) if content else None
                else:
                    frame = capture_from_ipcam(args.camera_url)
                if frame is None:
                    raise ValueError("Input image could not be decoded")
                warped, annotated, state, direction = analyze(frame, args.board_crop)
            except FrameUnavailable as error:
                if args.image or args.once:
                    raise
                print(f"Waiting for camera/board: {error}", file=sys.stderr, flush=True)
                if gui_initialized:
                    if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                           for name in ("2048 Board Detection", "2048 Vision Solver")):
                        break
                    waiting = np.zeros((600, 600, 3), dtype=np.uint8)
                    cv2.putText(waiting, "Waiting for camera/board", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    for name in ("2048 Board Detection", "2048 Vision Solver"):
                        cv2.imshow(name, waiting)
                    if (cv2.waitKey(1000) & 0xFF in (27, ord('q'))
                            or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                                   for name in ("2048 Board Detection", "2048 Vision Solver"))):
                        break
                else:
                    time.sleep(1)
                continue
            if gui_initialized and any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                                       for name in ("2048 Board Detection", "2048 Vision Solver")):
                break
            result = dict(grid=state.tolist(), direction=DIRECTIONS.get(direction),
                          game_over=direction is None, version=VERSION)
            if args.output_json:
                ensure_runtime_output(args.output_json)
            encoded = json.dumps(result)
            if args.output_json:
                previous_output = save_result(args.output_json, encoded, previous_output)
            print(encoded, flush=True)
            if args.speak:
                speak(DIRECTIONS[direction] if direction is not None else "game over")
            if not args.no_gui:
                if not gui_initialized:
                    open_windows()
                    gui_initialized = True
                cv2.imshow("2048 Board Detection", annotated)
                visualize_grid(warped, state, direction)
            if args.image or args.once:
                if not args.no_gui:
                    while cv2.waitKey(50) & 0xFF not in (27, ord('q')):
                        if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                               for name in ("2048 Board Detection", "2048 Vision Solver")):
                            break
                break
            if not args.no_gui:
                if (cv2.waitKey(1000) & 0xFF in (27, ord('q'))
                        or any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
                               for name in ("2048 Board Detection", "2048 Vision Solver"))):
                    break
            if args.no_gui:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

