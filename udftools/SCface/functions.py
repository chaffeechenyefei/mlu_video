

def decode_image_name(img_name):
    """
    {person_id}_cam{id}_{i-th}.jpg
    :param img_name: 
    :return: 
    """
    img_name = img_name.replace('.jpg','')
    ximg_name = img_name.split('_')
    person_id,cam_id = ximg_name[0:2]
    cam_id = int(cam_id.replace('cam',''))
    uuid = '%s_%d'%(person_id,cam_id)
    return {
        'person_id':person_id,
        'cam_id':cam_id,
        'uuid':uuid
    }

def decode_gallery_image_name(img_name:str):
    """
    {person_id}_frontal.JPG
    013_frontal.JPG
    :param img_name: 
    :return: 
    """
    img_name = img_name.lower()
    img_name = img_name.replace('.jpg','')
    person_id,_ = img_name.split('_')

    return {'person_id':int(person_id)}
