
def create_model(opt):
    model = None
    print(opt.model)

    # this is model use image information to compute the weight   	
    # not used anymore
    if opt.model == 'pix2pix_bucket':
        from .pix2pix_model_bucket import Pix2PixModel_Bucket
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket()

    # this is the end to end model	
    if opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        assert(opt.align_data == True)
        model = Pix2PixModel()
    # this is the bucket model based on two lstm + one lstm in discriminator	
    if opt.model == 'pix2pix_bucket2':
        from .pix2pix_model_bucket2 import Pix2PixModel_Bucket2
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket2()
    # this is the bucket model based on one lstm + one lstm in discriminator
    if opt.model == 'pix2pix_bucket3':
        from .pix2pix_model_bucket3 import Pix2PixModel_Bucket3
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket3()
    # this is the bucket model based on one lstm classification + one lstm in discriminator	
    if opt.model == 'pix2pix_bucket4':
        from .pix2pix_model_bucket4 import Pix2PixModel_Bucket4
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket4()
    # this is the bucket model based on one lstm classification + one lstm in discriminator	
    if opt.model == 'pix2pix_bucket5':
        from .pix2pix_model_bucket5 import Pix2PixModel_Bucket5
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket5()
    # this is the bucket model based on one lstm classification + one lstm in discriminator	
    if opt.model == 'pix2pix_bucket6':
        from .pix2pix_model_bucket6 import Pix2PixModel_Bucket6
        assert(opt.align_data == True)
        model = Pix2PixModel_Bucket6()
	
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
