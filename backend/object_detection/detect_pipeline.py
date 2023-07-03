import cv2
import yolov5
import json
import numpy as np
from object_detection.assign_teams import assign_teams

if __name__ == "__main__":
    # load model
    model = yolov5.load('keremberke/yolov5n-football')
    
    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # set image
    img = './example.png'

    # perform inference
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5] 
    # 0 - ball
    # 1 - player

    print('Printing first box prediction coords')
    print(boxes[0])

    # show detection bounding boxes on image
    results.show()

    # save results into "results/" folder
    results.save(save_dir='results/')

    def extract_ball_crds(boxes, categories):
        for i, box in enumerate(boxes):
            if categories[i] == 0:
                x1, y1, x2, y2 = box
                bottom_center = ((x1 + x2) // 2, y2)
                return bottom_center
    # throw error if no ball detected
    ball_crds = [extract_ball_crds(boxes, categories)]
    team1_crds, team2_crds = assign_teams(img, boxes, categories)

    # Convert any Tensor objects to numpy arrays fro JSON export
    ball_crds = [tuple(np.array(coord).tolist()) for coord in ball_crds]
    team1_crds = [tuple(np.array(coord).tolist()) for coord in team1_crds]
    team2_crds = [tuple(np.array(coord).tolist()) for coord in team2_crds]

    data = {
        'ball_crds': ball_crds,
        'team1_crds': team1_crds,
        'team2_crds': team2_crds
    }
    # Export coordinates to JSON
    with open('3D-coordinates.json', 'w') as file:
        json.dump(data, file)


def extract_and_assign_objects(image):
    # load model
    model = yolov5.load('keremberke/yolov5n-football')

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    # perform inference
    results = model(image, size=640)

    # inference with test time augmentation
    results = model(image, augment=True)

    # parse results
    predictions = results.pred[0] # 0 is ball and 1 is player
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5] 

    # save results into "results/" folder
    # results.save(save_dir='results/')

    def extract_ball_crds(boxes, categories):
        for i, box in enumerate(boxes):
            if categories[i] == 0:
                x1, y1, x2, y2 = box
                bottom_center = ((x1 + x2) // 2, y2)
                return bottom_center
        # throw error if no ball detected
    ball_crds = [extract_ball_crds(boxes, categories)]
    team1_crds, team2_crds = assign_teams(image, boxes, categories)

    # Convert any Tensor objects to numpy arrays fro JSON export
    ball_crds = [tuple(np.array(coord).tolist()) for coord in ball_crds]
    team1_crds = [tuple(np.array(coord).tolist()) for coord in team1_crds]
    team2_crds = [tuple(np.array(coord).tolist()) for coord in team2_crds]

    result = {
        'ball_crds': ball_crds,
        'team1_crds': team1_crds,
        'team2_crds': team2_crds
    }
    return result