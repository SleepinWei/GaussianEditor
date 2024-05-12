from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.scene.dataset_readers import KITTI360
import os
import json
import numpy as np 
from gaussiansplatting.scene.cameras import Simple_Camera 

class VastGaussianModelKITTI:
    def __init__(self,data_dir, seq=0,cam=0):
        # metadata chunk boundaries & chunk connections 
        # self.chunk_boundaries = {}  # chunk boundary infos 
        self.loaded_chunks = {} # (chunk_id : gaussian model)

        sequence_name = f"2013_05_28_drive_{seq:04d}_sync"
        self.chunk_root_path = os.path.join(data_dir,"chunks",sequence_name)
        self.metadata = KITTI360(data_dir=data_dir,seq=seq,cam=0) # a KITTI360 object from dataset_readers.py

        with open(os.path.join(self.chunk_root_path,"chunk_boundaries.json"),"r") as fp:
            self.chunk_boundaries = json.load(fp)
            self.chunk_ids = list(self.chunk_boundaries.keys())
            # 只比较 x,y 
            self.chunk_boundaries_min = np.vstack([np.array(v["min"][:2]) for k,v in self.chunk_boundaries.items()])
            self.chunk_boundaries_max = np.vstack([np.array(v["max"][:2]) for k,v in self.chunk_boundaries.items()])

        with open(os.path.join(self.chunk_root_path,"chunk_connection.json"),"r") as fp: 
            self.chunk_connection =  json.load(fp) # chunk_id : [connected chunks]
    
        self.colors = {chunk_id: np.random.randint(0,256,size=3) for chunk_id in self.chunk_boundaries.keys()}
        self.pc_for_render = GaussianModel(0,0,0,0)

    # current chunks
    # f: focal length 
    def get_current_chunks(self,_position:np.ndarray,f,width,height):
        # fint the chunk where the camera is 
        # position = cam.trans[:2] # x,y
        # position = c2w[:2,3]
        position = _position[:2]
        mask = np.logical_and(self.chunk_boundaries_min  < position, self.chunk_boundaries_max > position)
        indices = np.where(mask.sum(axis=1) == 2)
        cam_chunks = [self.chunk_ids[int(i)] for i in indices] 

        # find related chunks 
        result =  cam_chunks.copy()
        for cam_chunk in cam_chunks: 
            result.extend(self.chunk_connection[cam_chunk])
        result = list(set(result)) # unique
        return result
        # print(self.chunk_ids[indices]) # 满足两个条件

    def tick(self,position,f,width,height): # c2w,f,width,height): 
        current_chunks = self.get_current_chunks(position,f,width,height)
        refresh_pc_render = False 
        for chunk_id in current_chunks: 
            if chunk_id not in self.loaded_chunks:
                # load chunk 
                self.load_chunk(chunk_id)
                print(f"[INFO] {chunk_id} loaded")
                refresh_pc_render = True
        
        for k in list(self.loaded_chunks.keys()):
            if k not in current_chunks: 
                self.loaded_chunks.pop(k) # delete unnecessary chunks 
                print(f"[INFO] {k} off-loaded")
                refresh_pc_render = True

        if refresh_pc_render: 
            self.pc_for_render.from_models(self.loaded_chunks) # refresh chunks

    def load_chunk(self,chunk_id):
        print(f"[INFO] loading chunk {chunk_id}")
        chunk_dir = os.path.join(self.metadata.root_dir,"chunks",self.metadata.sequence_name,str(chunk_id))
        points_path = os.path.join(chunk_dir,"point_cloud.ply")

        # ZYW DEBUG
        if not os.path.exists(chunk_dir):
            print(f"[WARNING]{chunk_id} does not exist, use chunk_id=22")
            chunk_dir = os.path.join(self.metadata.root_dir,"chunks",self.metadata.sequence_name,str(22))
            points_path = os.path.join(chunk_dir,"point_cloud.ply")

        gaussian_model = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        gaussian_model.load_ply(points_path)
        self.loaded_chunks[str(chunk_id)] = gaussian_model

        

    def render(self,viewpoint_camera,pipe,background): 
        from gaussiansplatting.gaussian_renderer import render
        # render_results = []
        # for chunk_id,gauss in self.loaded_chunks.items():
        # render_pkg = render(cam, gauss, pipe, background, False)
        pc = self.pc_for_render
        # pc.from_models(self.loaded_chunks)
        return render(viewpoint_camera,pc,pipe,background)


if __name__ == "__main__":
    import math
    gaussian = VastGaussianModelKITTI("/DATA1/zhuyunwei/KITTI-360")
    c2w = np.array([[1,0,0,1120],
                    [0,1,0,3400],
                    [0,0,1,0],
                    [0,0,0,1]])
    f = 522.55
    width = 100
    height = 100
    fovx = 2 * math.atan(width / 2 /f)
    fovy = 2 * math.atan(height/ 2 /f)
    print(gaussian.get_current_chunks(c2w,f,width,height))