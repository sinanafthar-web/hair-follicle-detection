# Hair Follicle Segmentation & Triangle Detection App

A modular Streamlit application that uses Roboflow workflows for image segmentation to detect hair follicles and automatically generates triangular analysis with directional arrows. Built with a clean, modular architecture using the `inference-sdk` and environment-based configuration.

## Features

- 🔬 **Workflow-based Processing**: Uses Roboflow workflows for powerful image segmentation
- 📐 **Triangle Detection**: Finds the smallest triangle that encloses each segmented object
- ➡️ **Directional Analysis**: Calculates and displays arrows from the shortest triangle side to the apex
- 🎨 **Visual Overlay**: Draws triangles, arrows, and reference points on the processed image
- 📊 **Analysis Results**: Provides detailed statistics about detected objects
- 🔒 **Secure Configuration**: Environment-based configuration using `.env` files
- ⚡ **Performance Optimization**: Built-in workflow caching for faster processing

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your configuration:

```bash
# Copy the example and edit with your values
cp .env.example .env
```

Or create it manually:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace_name
ROBOFLOW_WORKFLOW_ID=your_workflow_id
CONFIDENCE_THRESHOLD=0.4
USE_CACHE=True
```

## Setup

### 1. Roboflow Account Setup

1. Create an account at [roboflow.com](https://roboflow.com)
2. Create a new project for hair follicle detection
3. Upload and label your hair follicle images
4. Create or use an existing workflow for segmentation
5. Get your credentials:
   - **API Key**: Found in your Roboflow account settings
   - **Workspace Name**: Your workspace identifier (e.g., `ranaudio`)
   - **Workflow ID**: Your workflow identifier (e.g., `small-object-detection-sahi`)

### 2. Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Step 1: Configure Environment
Set up your `.env` file with the required configuration:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace_name  
ROBOFLOW_WORKFLOW_ID=your_workflow_id
CONFIDENCE_THRESHOLD=0.4
USE_CACHE=True
```

### Step 2: Run the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

### Step 3: Upload and Process
1. Use the file uploader to select an image
2. Supported formats: PNG, JPG, JPEG, BMP, TIFF
3. Click the "Process Image" button
4. Wait for the analysis to complete
5. View the results in the processed image panel

The sidebar will display your current configuration loaded from the `.env` file.

## Example Usage

This app is based on the Roboflow inference-sdk workflow approach:

```python
from inference_sdk import InferenceHTTPClient
import supervision as sv

# Initialize client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="1SxAyEbpaNdwrNSbmDon"
)

# Run workflow
result = client.run_workflow(
    workspace_name="ranaudio",
    workflow_id="small-object-detection-sahi",
    images={
        "image": "your_image.jpg"
    },
    use_cache=True
)

# Load results into supervision Detections
detections = sv.Detections.from_inference(result)
```

## How It Works

### Image Processing Pipeline

1. **Segmentation**: The uploaded image is sent to your Roboflow model for segmentation
2. **Contour Extraction**: Segmentation masks are converted to contours
3. **Triangle Calculation**: For each contour, the algorithm finds the smallest triangle that completely encloses the object
4. **Direction Analysis**: The shortest side of each triangle is identified, and an arrow is drawn from its midpoint to the opposite apex

### Visual Elements

- **🟡 Yellow circles**: Middle points of shortest triangle sides
- **🔴 Red circles**: Triangle apex points  
- **🔵 Blue triangles**: Smallest enclosing triangles
- **🟢 Green arrows**: Direction from shortest side midpoint to apex
- **🟠 Cyan contours**: Original segmentation boundaries

### Algorithm Details

#### Triangle Detection
The app uses a brute-force approach to find the minimum area triangle:
1. Extracts the convex hull of each segmented object
2. Tests all combinations of 3 points from the convex hull
3. Validates that all original contour points lie within each candidate triangle
4. Selects the triangle with the minimum area

#### Direction Calculation
1. Calculates the length of all three triangle sides
2. Identifies the shortest side
3. Finds the midpoint of the shortest side
4. Determines the opposite apex (vertex not on the shortest side)
5. Draws an arrow from midpoint to apex

## Architecture

This application follows a modular architecture pattern:

- **`src/config.py`**: Centralized configuration management with constants and default settings
- **`src/triangle_detector.py`**: Core geometric algorithms for triangle detection and analysis
- **`src/image_processor.py`**: Image processing pipeline and Roboflow inference integration
- **`src/utils.py`**: Utility functions for drawing, image conversion, and validation
- **`app.py`**: Main Streamlit interface that orchestrates all components

This modular design provides:
- ✅ **Separation of Concerns**: Each module has a specific responsibility
- ✅ **Maintainability**: Easy to update individual components
- ✅ **Testability**: Components can be tested independently
- ✅ **Reusability**: Modules can be imported and used in other projects

## File Structure

```
hair-follicle-app/
├── src/                    # Source modules
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration constants
│   ├── image_processor.py  # Image processing and inference
│   ├── triangle_detector.py# Triangle detection algorithms
│   └── utils.py           # Utility functions
├── .env                   # Environment variables (create this file)
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── app.py                # Main Streamlit application
├── example_usage.py      # Example usage script
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

### Environment Variables Template

Create a `.env` file with the following structure:

```env
# Roboflow Configuration
ROBOFLOW_API_KEY=1SxAyEbpaNdwrNSbmDon
ROBOFLOW_WORKSPACE=ranaudio
ROBOFLOW_WORKFLOW_ID=small-object-detection-sahi

# Processing Configuration
CONFIDENCE_THRESHOLD=0.4
USE_CACHE=True
```

## Dependencies

- **streamlit**: Web app framework
- **inference-sdk**: Roboflow inference SDK for workflow deployment
- **supervision**: Computer vision utilities and annotations
- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations
- **Pillow**: Image processing
- **matplotlib**: Additional plotting capabilities
- **python-dotenv**: Environment variable management from `.env` files

## Troubleshooting

### Common Issues

**"Environment variable not found"**
- Ensure your `.env` file exists in the project root directory
- Check that all required variables are set in your `.env` file
- Restart the Streamlit app after modifying the `.env` file
- Verify there are no syntax errors in your `.env` file (no spaces around `=`)

**"No segmentation masks found in detections"**
- Ensure your workflow includes segmentation capabilities
- Check that your workflow is properly configured and deployed
- Verify the image contains objects your workflow can detect

**"Error processing image"**
- Check your internet connection
- Verify your workspace name and workflow ID are correct in the `.env` file
- Ensure your workflow is accessible with your API key
- Check that the workflow supports the type of analysis you're trying to perform

**Triangles not appearing**
- This may happen if the segmented objects are too small or simple
- Try with images that have more complex, larger objects

### Performance Tips

- Use images with resolution between 416x416 and 1024x1024 for best results
- Ensure good lighting and contrast in your images
- Configure your Roboflow workflow with appropriate parameters for your use case
- Enable caching to improve performance on repeated workflow executions

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 