# 2048 Vision Solver

Recover tile values from an image of a 4×4 2048 board and get a legal move recommendation from expectimax. The solver displays the detected board and grid and can speak the move on Windows.

## Install

Requires Windows 10/11, Python 3.12 and [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) with English (`eng`) language data.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Use `--tesseract-cmd` to select your installed Tesseract executable.

## Read an image

Pass a screenshot or photo of your 2048 board.

```powershell
$imagePath = Read-Host 'Path to the board image'
.venv\Scripts\python main.py --image "$imagePath" --output-json result.json
```

Choose a new result filename or pass `--overwrite-output` to replace an existing file.

By default the solver finds the board corners and corrects perspective. Use `--board-crop` when the image already contains only the full board.

Close the grid window or press **Escape** or **Q** in it to stop. Use `--no-gui` to hide windows and `--speak` for Windows SAPI output.

## IP webcam

Set `--camera-url` to your camera's snapshot URL.

```powershell
$cameraUrl = Read-Host 'IP camera snapshot URL'
.venv\Scripts\python main.py --camera-url "$cameraUrl" --speak
```

`--once` reads a single snapshot.

Close the grid window, press **Escape** or **Q** in it, or press **Ctrl+C** in the terminal to stop.

JSON output includes `grid`, `direction` (`left`, `up`, `right`, `down`) and `game_over`. A board with no legal move returns a null direction.

With `--output-json`, continuous capture updates the result for each recognized frame.

Keep the full board visible and use large cells with readable digits.
