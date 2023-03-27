import os 

if __name__ == '__main__':
    dest = '/scratch/hong_seungbum/datasets/MLD/inpainting/v4/flist'
    for item_type in ["ball", "lid"]:
        for train_val in ["train", "val"]:
            for img_mask in ["img", "mask"]:
                path = f"/scratch/hong_seungbum/datasets/MLD/inpainting/v4/{img_mask}/{item_type}/{train_val}"
            
                with open(os.path.join(dest,f"{item_type}_{img_mask}_{train_val}.flist"), 'w') as f:
                    for filename in sorted(os.listdir(path)):
                        f.write(f"{os.path.join(path, filename)}\n")
            