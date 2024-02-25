from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import os
from fastapi.responses import JSONResponse
from pathlib import Path
from multiprocessing import Value
from io import BytesIO
from PIL import Image
from base64 import b64encode
from PIL import Image, UnidentifiedImageError

from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import os

app = FastAPI()

# Initialize counts for each category
car_count = 0
truck_count = 0
bus_count = 0

# Keep track of the identifier of detected objects (ids)
detected_obj_ids = {
    'car': [],
    'truck': [],
    'bus': []
}


def save_img(img):
    img_dir = "esp32_imgs"
    if not Path(img_dir).is_dir():
        Path(img_dir).mkdir()
    cv2.imwrite(os.path.join(img_dir, f"input.jpg"), img)


def process_video(wf, tracking, threshold, drop):
    global car_count, truck_count, bus_count, detected_obj_ids

    print("process_video")

    # Read image from stream
    input_img_path = str(Path.cwd().joinpath(
        "esp32_imgs", "input.jpg"))
    frame = cv2.imread(input_img_path)
    if frame is None:
        print("Error: Unable to load the image")
        return 0, 0, 0  # Return default values or handle the error appropriately

    # Run the workflow on the current frame
    wf.run_on(array=frame)

    # Get results
    image_out = tracking.get_output(0)
    obj_detect_out = tracking.get_output(1)

    # Convert the result to BGR color space for displaying
    img_out = image_out.get_image_with_graphics(obj_detect_out)
    img_res = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    # ________________________   start C O U N T I N G

    # Get the detected objects in the frame
    detected_objects = obj_detect_out.get_objects()
    print("detected_objs = ", detected_objects)

    # Loop through detected objects and update the class counts
    # The same object is not counted simultanisly= tracked by its id
    # The class count is updated once the object class surpasses the trust line
    for obj in detected_objects:
        # print(obj)
        label = obj.label
        ident = obj.id

        if (label == "car") and (ident not in detected_obj_ids["car"]):
            detected_obj_ids["car"].append(ident)  # keep track
            car_count += 1
        elif (label == "truck") and (ident not in detected_obj_ids["truck"]):
            detected_obj_ids["truck"].append(ident)  # keep track
            truck_count += 1
        elif (label == "bus") and (ident not in detected_obj_ids["bus"]):
            detected_obj_ids["bus"].append(ident)  # keep track
            bus_count += 1

    # Flush the tracking buffer if the threshold is reached
    detected_obj_ids = {k: v[drop:]
                        if len(v) > threshold else v
                        for k, v in detected_obj_ids.items()}

    # ________________________   end C O U N T I N G

    # Save the resulting frame
    output_img_path = str(Path.cwd().joinpath(
        "esp32_imgs", "output.jpg"))
    cv2.imwrite(output_img_path, img_out)

    print(f'output saved with cars={car_count}, trucks={truck_count}, bus={bus_count}\n')

    return 0



@app.get("/stream")
def get_image():
    while True:
        try:
            with open("image.jpg", "rb") as f:
                image_bytes = f.read()
            image = Image.open(BytesIO(image_bytes))
            img_io = BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            img_bytes = img_io.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        except Exception as e:
            print("encountered an exception: ")
            print(e)

            with open("placeholder.jpg", "rb") as f:

                image_bytes = f.read()
            image = Image.open(BytesIO(image_bytes))
            img_io = BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            img_bytes = img_io.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
            continue


@app.post("/upload")
async def upload(imageFile: UploadFile = File(...)):
    try:
        print("processing")
        contents = await imageFile.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        save_img(img)
        #frame_setup(img)

        # launch processing
        process_video(wf, tracking, threshold, drop)
        
        global car_count, truck_count, bus_count, detected_obj_ids
        print("cars: ", car_count, "trucks: ",truck_count, "buses: ", bus_count)
        print("success")
        return JSONResponse(content={"message": "Image Received"}, status_code=201)
    except Exception as e:
        return HTTPException(detail=str(e), status_code=500)


if __name__ == "__main__":
    import uvicorn

    # ____________________ Model Initialisation

    # Init your workflow
    wf = Workflow()

    # Add object detection algorithm
    detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

    # Add ByteTrack tracking algorithm
    tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

    tracking.set_parameters({
        "categories": "car,truck,bus",  # List of object classes
        "conf_thres": "0.7",   # Increase this parameter for more accuracy
    })

    # Set a limit to flush the ids tracking buffer => for performancee issues
    threshold = 10
    # The buffer pourcentage to flush - from the beginning
    drop = threshold//3

    uvicorn.run(app, host="0.0.0.0", port=8000)