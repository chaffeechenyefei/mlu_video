"""
moving and renaming for CMU-PIE
--data cmu-pie
cmu-pie/{sub_batch_folder}/{person_id}/{expression/illumination}/*.jpg
"""

import argparse,os,shutil

pj = os.path.join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--save')
    parser.add_argument('--ext',default='.jpg')
    parser.add_argument('--subfolder',default='expression')
    args = parser.parse_args()

    print('This program will execute %s and put all the images into one folder in %s'%(args.data,args.save))

    if not os.path.exists(args.save):
        os.mkdir(args.save)

    sub_batch_folders = [ c for c in os.listdir(args.data) if os.path.isdir(pj(args.data,c)) ]

    for cnt_sub_batch_folder,sub_batch_folder in enumerate(sub_batch_folders):
        print('Executing %s(%d/%d)'%(sub_batch_folder,cnt_sub_batch_folder+1,len(sub_batch_folders)))
        sub_batch_folder_full = pj(args.data,sub_batch_folder)
        person_folders = [ c for c in os.listdir(sub_batch_folder_full) if os.path.isdir(pj(sub_batch_folder_full,c))]

        for cnt_person_folder, person_folder in enumerate(person_folders):
            print('-->%s(%d/%d)'%(person_folder,cnt_person_folder+1,len(person_folders)))
            person_folder_full = pj(sub_batch_folder_full,person_folder)
            subfolder_full = pj(person_folder_full,args.subfolder)
            if not os.path.isdir(subfolder_full):
                print('%s/%s is empty'%(person_folder,args.subfolder))
                continue

            img_names = [ c for c in os.listdir(subfolder_full) if c.endswith(args.ext) ]

            for cnt_img,img_name in enumerate(img_names):
                print('%s(%d/%d)'%(img_name,cnt_img+1,len(img_names)),end='\r')
                img_name_full = pj(subfolder_full,img_name)
                save_img_name = '%s_%s'%(person_folder,img_name)
                save_img_name_full = pj(args.save,save_img_name)

                shutil.copyfile(img_name_full,save_img_name_full)

    print('END')

