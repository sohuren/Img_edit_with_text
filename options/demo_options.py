from .base_options import BaseOptions

class DemoOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--image', type=str, default='', help='the image path')
	self.parser.add_argument('--saved_image', type=str, default='', help='the saved image path')
	self.parser.add_argument('--description', type=str, default='', help='the description that you want to apply')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        
	self.isTrain = False
