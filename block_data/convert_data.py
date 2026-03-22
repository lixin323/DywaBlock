import os  
import numpy as np  
import trimesh  
from pkm.data.transforms.sample_points_from_urdf import sample_surface_points_from_urdf  

def convert_obj_to_corn(obj_path, object_name, output_dir="/input/DGN"):  
    """将OBJ文件转换为CORN格式"""  
      
    # 1. 创建目录结构  
    os.makedirs(f"{output_dir}/meta-v8/cloud", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/cloud-2048", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/normal", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/normal-2048", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/urdf", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/meta", exist_ok=True)  
    os.makedirs(f"{output_dir}/meta-v8/hull", exist_ok=True)  
    os.makedirs(f"{output_dir}/coacd", exist_ok=True)  
      
    # 2. 创建URDF文件  
    urdf_content = f'''<?xml version="1.0"?>  
<robot name="{object_name}">  
  <link name="base_link">  
    <visual>  
      <geometry>  
        <mesh filename="{object_name}.obj"/>  
      </geometry>  
    </visual>  
    <collision>  
      <geometry>  
        <mesh filename="{object_name}.obj"/>  
      </geometry>  
    </collision>  
    <inertial>  
      <mass value="0.1"/>  
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>  
    </inertial>  
  </link>  
</robot>'''  
      
    urdf_path = f"{output_dir}/meta-v8/urdf/{object_name}.urdf"  
    with open(urdf_path, 'w') as f:  
        f.write(urdf_content)  
      
    # 复制OBJ文件  
    import shutil  
    shutil.copy(obj_path, f"{output_dir}/meta-v8/urdf/{object_name}.obj")  
      
    # 3. 生成点云和法向量  
    from pathlib import Path  
    urdf_file = Path(urdf_path)  
      
    # 生成2048点点云  
    sample_surface_points_from_urdf(  
        filename=urdf_file,  
        count=2048,  
        use_poisson=True,  
        use_even=True,  
        cloud_dir=f"{output_dir}/meta-v8/cloud-2048/",  
        normal=True,  
        normal_dir=f"{output_dir}/meta-v8/normal-2048/",  
        export=True,  
        cat=False  
    )  
      
    # 生成原始点云（更多点）  
    sample_surface_points_from_urdf(  
        filename=urdf_file,  
        count=10000,  
        use_poisson=True,  
        use_even=True,  
        cloud_dir=f"{output_dir}/meta-v8/cloud/",  
        normal=True,  
        normal_dir=f"{output_dir}/meta-v8/normal/",  
        export=True,  
        cat=False  
    )  
      
    # 4. 生成凸包  
    mesh = trimesh.load(obj_path)  
    hull = mesh.convex_hull  
    hull.export(f"{output_dir}/meta-v8/hull/{object_name}.obj")  
      
    print(f"转换完成：{object_name}")  
  
# 使用示例  
convert_obj_to_corn("your_object.obj", "your_object")