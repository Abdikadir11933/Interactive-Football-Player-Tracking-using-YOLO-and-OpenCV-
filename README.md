Interactive Football Player Tracking & Tactical Analysis App
A Streamlit-based application that uses computer vision to analyze football match footage, providing tactical insights through player tracking, heatmaps, pass networks, and team shape visualizations.




Description
This project is a Football Analysis Suite designed for coaches, analysts, and players to gain deeper tactical insights from match videos. The application processes video footage to track player and ball movements, transforms this data onto a real-world pitch map, and generates interactive visualizations. The primary goal is to move beyond simple statistics and provide a visual understanding of player positioning, spatial awareness, pass combinations, and team formations.

The tool is especially useful for:

Midfielders and Strikers looking to analyze their positioning and off-the-ball movement.
Coaches and Analysts who need objective, data-driven tools to assess player performance and team tactics.
Table of Contents
Features
Technology Stack
Installation
Usage
Contributing
License
Contact
Features
Video Analysis Engine: Upload a video clip, trim it to a specific segment, and let the app process it.
Player & Ball Tracking: Utilizes the YOLOv8 model for high-performance object detection and tracking on a frame-by-frame basis.
Homography Transformation: Maps raw video pixel coordinates to real-world meter coordinates on a standardized football pitch, enabling accurate spatial analysis.
Interactive Setup: Features an interactive canvas to easily map reference points from the video to the pitch diagram for an accurate transformation.
Contextual Heatmaps: Generate player heatmaps based on specific tactical situations, such as positioning during pass receptions or in the attacking third.
Progressive Pass Networks: Visualize pass combinations as a tactical map, showing players at their average positions and color-coding passes by direction (e.g., forward, backward).
Team Shape Analysis: Overlays a convex hull on each team to visualize their defensive/offensive shape, compactness, and the space between the lines.
Frame-by-Frame Viewer: Isolate and view the movement paths and trails of specific players on a clean pitch diagram to analyze their runs and patterns.
Player Statistics: Calculates basic metrics like estimated speed and distance covered for each tracked player.
Technology Stack
Language: Python
Framework: Streamlit
Computer Vision: OpenCV, Ultralytics (for YOLOv8)
Data Manipulation: NumPy, Pandas, SciPy
Visualization: Matplotlib, mplsoccer, Streamlit-Drawable-Canvas
Video Processing: FFmpeg
Installation
To set up the project locally, follow these steps:

Clone the repository:
Bash

git clone https://github.com/your-username/your-project.git
cd your-project
Install dependencies: It is recommended to create a virtual environment first.
Bash

pip install -r requirements.txt
Usage
To launch the application, run the following command from the root directory of the project:

Bash

streamlit run main_app.py
Navigate to the local URL provided by Streamlit in your web browser to start using the app.

Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

License
This project is licensed under the MIT License. See the LICENSE.md file for details.

Contact
Your Name - @YourTwitterHandle - your.email@example.com

Project Link: https://github.com/your-username/your-project
Vie Sheetsiinbeing ashes
License
This project is licensed under the MIT License. See the LICENSE file for details.
