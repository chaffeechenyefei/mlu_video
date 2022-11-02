model_dict = {
    # 'resnet101_irse_mx':{'path':'weights/face_rec','weights':'r101irse_model_3173.pth'},
    'resnet101_irse_mx':{'path':'weights/face_rec','weights':'r101irsemx_webface260m_model_1277.pth', 'no_serial_weights':'r101irsemx_webface260m_model_1277_no_serial.pth'},
    'resnet50_irse_mx': {'path':'weights/face_rec','weights':'r50irsemx_webface260m_model_539.pth', 'no_serial_weights':'r50irsemx_webface260m_model_539_no_serial.pth'},
    'baseline5932'   : {'path':'weights/face_rec'  ,'weights':'model_5932.pth'},
    'combine7628'    : {'path':'weights/face_rec'  ,'weights':'model_7628.pth'},
    'multihead7513'  : {'path':'weights/face_rec'  ,'weights':'model_7513.pth'},
    'compress_multihead7513'  : {'path':'weights/face_rec'  ,'weights':'compress_model_7513.pth'},
    'compress11757':{'path':'weights/face_rec'  ,'weights':'compress_model_11757.pth'},
}