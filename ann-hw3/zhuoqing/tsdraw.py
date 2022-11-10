from PIL import Image
import sys
import os

class PlotGIF(object):
    def __init__(self, gifdir=r'd:\temp\GIF'):
        self.gifdir = gifdir
        self.imageDim = []
        self.count = 0

        if os.path.isdir(gifdir) == False:
            os.mkdir(gifdir)

    def infor(self, ):
        print(self.gifdir, self.count)

    def __str__(self, ):
        return 'gifdir:%s' % self.dir

    def append(self, plt):
        filename = os.path.join(self.gifdir, '%04d.jpg' % self.count)
        self.count = self.count + 1
        plt.savefig(filename)
        self.imageDim.append(Image.open(filename))

    def appendbox(self, boxid, side=0):
        filename = os.path.join(self.gifdir, '%04d.jpg' % self.count)
        self.count = self.count + 1
        tspsaveimagebox(boxid, filename, side)
        self.imageDim.append(Image.open(filename))

    def appendrange(self, rangeid, side=0):
        filename = os.path.join(self.gifdir, '%04d.jpg' % self.count)
        self.count = self.count + 1
        tspsaveimagerange(rangeid, filename, side)
        self.imageDim.append(Image.open(filename))

    def save(self, giffile=r'd:\temp\gif1.gif', period=100, last=100):
        duration = [period] * len(self.imageDim)
        duration[-1] = last

        self.imageDim[0].save(giffile,
                              save_all=True,
                              append_images=self.imageDim[1:],
                              duration=duration,
                              loop=False)
        self.imageDim = []
        self.count = 0

        file_stats = os.stat(giffile)
        print('%s size:%5.2fM\a' %
               (giffile, file_stats.st_size / (1000*1000.0)))


def dop2gif(giffile=r'd:\temp\gif1.gif', gifid=0, period=50, last=50):
    dopfilename = tspgetresourcefile(gifid)
    if os.path.isfile(dopfilename) == False:
        return 1

    filedir = os.path.dirname(dopfilename)
    filedim = os.listdir(filedir)

    imageDim = []
    for f in filedim:
        imageDim.append(Image.open(os.path.join(filedir, f)))

    duration = [period] * len(imageDim)
    duration[-1] = last

    imageDim[0].save(giffile,
                     save_all=True,
                     append_images=imageDim[1:],
                     duration=duration,
                     loop=False)

    print('Save GIF: %s' % giffile)
    return 0


def dir2gif(giffile=r'd:\temp\gif1.gif', gifdir=r'd:\temp\gif', period=50, last=50):
    filedir = gifdir
    filedim = os.listdir(filedir)

    if len(filedim) == 0:
        return 0

    imageDim = []
    for f in filedim:
        imageDim.append(Image.open(os.path.join(filedir, f)))

    duration = [period] * len(imageDim)
    duration[-1] = last

    imageDim[0].save(giffile,
                     save_all=True,
                     append_images=imageDim[1:],
                     duration=duration,
                     loop=False)

    print('Save GIF: %s' % giffile)
    return 0


def filedim2gif(giffile=r'd:\temp\gif1.gif', filedim=[], period=50, last=50):
    if len(filedim) == 0:
        return 0

    imageDim = []
    for f in filedim:
        imageDim.append(Image.open(os.path.join(filedir, f)))

    duration = [period] * len(imageDim)
    duration[-1] = last

    imageDim[0].save(giffile,
                     save_all=True,
                     append_images=imageDim[1:],
                     duration=duration,
                     loop=False)

    print('Save GIF: %s' % giffile)
    return 0


def files2gif(giffile=r'd:\temp\gif1.gif', filename=r'd:\temp\gif\0000.jpg', number=0, period=50, last=50):
    if len(filename) == 0:
        return 0
    if number == 0:
        return 0

    removeflag = 0
    if number < 0:
        number = -number
        removeflag = 1

    filedir = os.path.dirname(filename)
    fileext = filename.split('.')[-1]
    imageDim = []

    for i in range(number):
        files = os.path.join(filedir, '%04d.%s' % (i, fileext))
        imageDim.append(Image.open(files))
        if removeflag == 1:
            os.remove(files)

    durations = [period] * len(imageDim)
    durations[-1] = last

    imageDim[0].save(giffile, save_all=True,
                     append_images=imageDim[1:],
                     duration=durations,
                     loop=False)

    print('Save GIF: %s' % giffile)
    return 0


def tempgifdir():
    gifdir = r'd:\temp\GIF'
    if os.path.isdir(gifdir) == False:
        os.mkdir(gifdir)
        return gifdir

    files = os.listdir(gifdir)
    for f in files:
        if os.path.isfile(f):
            os.remove(os.path.join(gifdir, f))

    return gifdir
