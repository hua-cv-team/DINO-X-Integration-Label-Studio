import os
import json
import tkinter.filedialog

try:
    from pycocotools.mask import decode
except ImportError:
    os.system("pip install pycocotools")
    from pycocotools.mask import decode

try:
    from dds_cloudapi_sdk import Config, Client
    from dds_cloudapi_sdk.image_resizer import image_to_base64
    from dds_cloudapi_sdk.tasks.v2_task import V2Task
except ImportError:
    os.system("pip install dds-cloudapi-sdk --upgrade")
    from dds_cloudapi_sdk import Config, Client
    from dds_cloudapi_sdk.image_resizer import image_to_base64
    from dds_cloudapi_sdk.tasks.v2_task import V2Task

# keys & values should be lowercase
MAPPING = {
    
    "car": "civilian vehicle",
    "van": "civilian vehicle",
    "bus": "civilian vehicle",
    "truck": "civilian vehicle",
    "motorcycle": "civilian vehicle",
    "boat": "civilian vehicle",
    "bicycle": "civilian vehicle",
    "minivan": "civilian vehicle",
    "nissan": "civilian vehicle",
    "vehicle": "civilian vehicle",
    "ship": "civilian vehicle",
    "bike": "civilian vehicle",
    "jeep": "civilian vehicle",
    "scooter": "civilian vehicle",
    "taxi": "civilian vehicle",
    "suv": "civilian vehicle",
    
    "bird": "animal",
    "dog": "animal",
    "mammal": "animal",
    "chicken": "animal",
    "elephant": "animal",
    "giraffe": "animal",
    "goose": "animal",
    "horse": "animal",
    "sheep": "animal",
    "animal": "animal",
    "fish": "animal",
    "duck": "animal",
    "pig": "animal",
    "bear": "animal",
    
    "person": "citizen",
    "pedestrian": "citizen",
    "woman": "citizen",
    "people": "citizen",
    
    "mound": "water",
    "pool": "water",
    "lake": "water",
    "river": "water",
    "puddle": "water",
    
    "flower": "plant",
    
    "house": "building",
    "skyscraper": "building",
    "bridge": "building",
    
    "hurdle": "barrier",
    
    "garbage": "debris",
    "rubble": "debris",
    
    "bulldozer": "excavator",
    
    "glasses": "protective glasses",
    
    "sneaker": "boot", 
    "footwear": "boot",
    
    "handbag": "bag",
    "backpack": "bag",

    "hose": "fire hose",
    
    "fire": "flame",

    "sinkhole": "hole in the ground",
    "manhole": "hole in the ground",

}

def call_prompt_free_api(token:str, image_path:str, format:str="coco_rle", bbox_threshold:float=0.25, iou_threshold:float=0.8):
    client = Client(Config(token))
    image = image_to_base64(image_path)
    api_path="/v2/task/dinox/detection"
    api_body={
        "model": "DINO-X-1.0",
        "image": image,
        "prompt": {
            "type": "universal"
        },
        "targets": ["bbox", "mask"],
        "mask_format": format,
        "bbox_threshold": bbox_threshold,
        "iou_threshold": iou_threshold
    }
    task = V2Task(
        api_path=api_path,
        api_body=api_body
    )
    client.run_task(task)
    return task.result

def map_dinox_labels_to_TRIFFID(seg_result:dict, mapping:dict=MAPPING): # pass by reference
    for obj in seg_result["objects"]:
        if obj["category"].lower() in mapping.keys():
            obj["category"] = mapping[obj["category"].lower()].lower()
        else:
            obj["category"] = obj["category"].lower()
    return seg_result

def decode_RLE(seg_result:dict): # pass by reference
    for idx in range(len(seg_result["objects"])):
        seg_result["objects"][idx] = decode(seg_result["objects"][idx]["mask"]).tolist()
    return seg_result

def save_json(seg_result:dict, destination:str, indent:int|str|None=4):
    # if os.path.exists(destination):
    #     ans = input(f"{destination} already exists, so it will be overriden - are you sure you want to continue? (Y/N)").lower()
    #     if ans in ["no", "n"]:
    #         new_dst = input("Ok, enter other destination: ")
    #         save_json(seg_result=seg_result,destination=new_dst)
    with open(destination,"w") as out_file:
        json.dump(seg_result,out_file,indent=indent)

def ask_images():
    return tkinter.filedialog.askopenfilenames(parent=None,filetypes=[("Images","*.jpg *.jpeg *.jpe *.jfif *.jfi *.png *.gif *.bmp *.webp *.tiff *.tif")])

#####################################################################

MY_TOKEN = "YOUR_TOKEN"
DECODE_RLE = False

#####################################################################

if __name__ == "__main__":
    for path in ask_images():
        seg_res = call_prompt_free_api(token=MY_TOKEN,image_path=path)
        if DECODE_RLE:
            seg_res = decode_RLE(seg_result=seg_res)
        seg_res = map_dinox_labels_to_TRIFFID(seg_result=seg_res)
        save_json(seg_result=seg_res,destination=path+"_dinox_preannot.json")
