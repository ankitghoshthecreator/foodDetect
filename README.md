# YOLO PyCharm Project

## Setup
1. Open this folder in PyCharm.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run detection:
   ```bash
   python scripts/yolo_detect.py --model models/my_model.pt --source path/to/image_or_video
   ```

## Contents
- **models/** → Trained YOLO models (best.pt, last.pt, my_model.pt)
- **scripts/** → YOLO detection scripts
- **results/** → Training results and evaluation graphs
- **requirements.txt** → Dependencies

🚀 Happy Coding!
