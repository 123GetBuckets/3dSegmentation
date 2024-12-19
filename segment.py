#!/usr/bin/env python
#
# file: $NEDC_NFC/util/python/nedc_bio_3d_segment/nedc_bio_3d_img.py
#
# revision history:
#
# 20241111 (BR): implemented is_3dtiff function
# 20240818 (HK): initial version
#
# usage:
#  import nedc_bio_3d_img as nb3di
#
# description: {FILL LATER}

#------------------------------------------------------------------------------

# import required system modules
#
import os
import sys
import numpy as np
# contact erriele/import 3rd-party modules
import tifffile as tfl
import cv2 as cv
from PIL import Image as pil
import scipy

# import required nedc modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft

#------------------------------------------------------------------------------
#
# define important constants
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define the version
#
# {HERE}

# define paramter file constants
#
PARAM_MAP = "MAP"
NEDC_RED = "NEDC_RED"
NEDC_EDNA = "NEDC_EDNA"
NEDC_PUNCTA = "NEDC_PUNCTA"

# define paramter file constants
#
# {HERE}

# define error types
#
# {HERE}

#------------------------------------------------------------------------------
#
# classes listed here
#
#------------------------------------------------------------------------------

def process(ofile, ifile, params, ftype, analysis):
    # process an image
    #
    #>>> call the function >>>
    img3d = NedcBioImg(ifile, params[analysis])
    param = params[analysis]
    print(param)
    if analysis == NEDC_RED:
        # get morph close radius from params
        #
        r = param.get('close', 0)
        if r = 0:
            print("Error: %s (line: %s) %s: %s %s %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "error loading the ", n3dit.NEDC_RED, "param", "close"))

        # morph close on each slice
        #
        for i in range(img3d.nSlices):
            img3d.img[i] = scipy.ndimage.morphology.grey_closing(img3d.img[i], size=( 1+(r*2), 1+(r*2) ))
        img3d.segment(0, int(param['threshold']))

    # write segment tif file
    #
    if ftype == nft.DEF_EXT_TIF:
        img3d.writeTiff(ofile)
        print("     %s" % (ofile))
    else:
        print("Writing %s" % (ofile))
        ofile = nft.make_fp(ofile)
        if ofile == None:
            return None
        img3d.writeVoxelList(ofile)
    return True

#
# end of method

def is_3dtiff(fname, dbgl_d):
    """
    method: is_3dtiff

    arguments:
     fname: input filename

    returns:
     Returns a boolean value indicating if it was a 3D tiff file.
    """

    # display informational message
    #
    if dbgl_d > ndt.BRIEF:
        print("%s (line: %s) %s::%s: opening %s file" %
               (__FILE__, ndt.__LINE__, NedcBioImg.__CLASS_NAME__, ndt.__NAME__, fname))

    # check if it is a tiff file:
    #
    try:
        img = pil.open(fname)
        # iterate through stack, keep count of frames
        #
        if img.format == 'TIFF':
            frames = 1
            while True:
                try:
                    img.seek(img.tell() + 1)
                    frames += 1
                # reached end of stack
                #
                except EOFError:
                    if frames < 2:
                        #exit ungracefully: it is not 3D
                        #
                        return False
                    break
            img.close()
        else:
            #exit ungracefully: it is not a tiff file
            #
            return False
    except pil.UnidentifiedImageError:
        if dbgl_d > ndt.BRIEF:
            print("%s (line: %s) %s::%s: Unable to identify file: %s" %
                  (__FILE__, ndt.__LINE__, NedcBioImg.__CLASS_NAME__, ndt.__NAME__, fname))
        return False
    except Exception:
        return False
    #exit gracefully: it is a 3D tiff file!
    #
    return True
#
# end of method


class NedcBioImg():
    """
    Class: {name not decided}

    arguments:
     none

    description:
     {FILL LATER}
    """

    #--------------------------------------------------------------------------
    #
    # static data declarations
    #
    #--------------------------------------------------------------------------

    # define static variables for debug and verbosity
    #
    dbgl_d = ndt.Dbgl()
    vrbl_d = ndt.Vrbl()

    def __init__(self, ffile, params=None):
        """
        method: constuctor

        arguments:
         ffile: file pointer to the tiff file
         params: paremeters from the params file

        return:
         none

        description:
         initiate 3dimage class
        """
        # create class data
        #
        NedcBioImg.__CLASS_NAME__ = self.__class__.__name__

        # initialize tiff file
        #
        tiff = tfl.TiffFile(ffile)

        # initialize image array
        #
        self.img = tfl.imread(ffile)
        self.img = self.img.astype('uint16')

        # initialize metadata
        #
        self.metadata = {}
        tags = getattr(tiff.pages[0], 'tags', None)
        assert tags is not None
        for tag in tags.values():
            self.metadata[tag.name] = tag.value

        # initialize image dimensions
        #
        self.nSlices = len(tiff.pages)
        self.width = self.metadata['ImageWidth']
        self.length = self.metadata['ImageLength']
        #self.ch = params{'channel'}

        # Img ID for containing segmented image
        #  contains padding to allow segmenter to work without indexing issues
        #
        self.imgID = np.zeros(((self.nSlices + 1), (self.length + 2), (self.width + 2)), dtype='uint16')

        # init temp voxel dict
        #
        self.vdict = {}

        # mask for relative voxel position
        #  [z, y, x]
        #
        self.dtree = [
           [0, 0, 0],
            [0, 0, -1], [0, -1, -1], [0, -1, 0], [0, -1, 1],
            [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
            [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
            [-1, 1, -1], [-1, 1, 0], [-1, 1, 1]
        ]

        # optimized order for checking anterior voxels
        #  used for decision tree
        self.d_order = [9,3,6,1,8,10,2,5,12,4,7,11,13]

        # decision tree function list
        #
        self.decision_tree = [
            self.dtree1,
            self.dtree2,
            self.dtree3,
            self.dtree4,
            self.dtree5,
            self.dtree6,
            self.dtree7,
            self.dtree8,
            self.dtree9,
            self.dtree10,
            self.dtree11,
            self.dtree12,
            self.dtree13
        ]

        # provisional label list
        #
        self.plabel = []
        self.plabel.append(None)

        # representative label list
        #
        self.rlabel = []
        self.rlabel.append(0)

    #
    # *incomplete*


    def threshold(self, val, thresh):
        """
        method: threshold

        arguments:
         val: value of voxel

        return: threshold value or 0

        description:
         this function is a wrapper for multiple thresholding
         strategies
        """

        # simple version
        #
        if (val < thresh):
            return 0

        return thresh

    #
    # end of method


    def segment(self, channel, thresh):
        """
        method: segment

        return: none

        description:
         this function connects thresholded pixels within
         a certain radius together.
        """

        clr = channel
        # initialize current tag
        #
        nextTag = 0

        # initialize minimum tag
        #
        minTag = 0
        rlabel_len = 0

        # step 1: connecting structures
        #
        for z in range(1, self.nSlices + 1):
            # display informational message
            #
            if self.dbgl_d > ndt.MEDIUM:
                print(str(z) + ": Slice init tag complete")

            for y in range(1, self.length + 1):

                for x in range(1, self.width + 1):
                    # increment nextTag if previous voxel was isolated
                    #
                    if minTag == nextTag:
                        nextTag += 1

                    # if voxel is above threshold, set minTag to nextTag
                    #
                    if (val := self.threshold(self.img[z - 1, y - 1, x - 1], thresh)) > 0:
                        minTag = nextTag
                        for m in self.d_order:

                            # get position of the voxel that is being checked
                            #
                            pt = np.add([z,y,x], self.dtree[m])

                            # run decision tree function for the first voxel found
                            #
                            if self.imgID[pt[0], pt[1], pt[2]] > 0:
                                #  m - 1 because there is no 0 in the function list
                                #
                                minTag = self.decision_tree[m-1]([z,y,x])
                                break

                        # handle if the tag chosen is new
                        #
                        if minTag > rlabel_len:
                            self.rlabel.append(minTag)
                            rlabel_len += 1
                            self.plabel.append(set([minTag]))

                        # finally update imgID
                        #
                        self.imgID[z,y,x] = self.rlabel(minTag)
        #
        # initial tagging and label resolution complete

        # apply the label equivalencies and populate voxel dict
        #
        for z in range(1, self.nSlices + 1):
            for y in range(1, self.length + 1):
                for x in range(1, self.width + 1):
                    self.imgID[z,y,x] > 0:
                        self.imgID[z,y,x] = self.rlabel[self.imgID[z,y,x]]
                        if self.imgID[z,y,x] in self.vdict:
                            self.vdict[self.imgID[z,y,x]].append((x, y, z))
                        else:
                            self.vdict[self.imgID[z,y,x]] = [(x,y,z)]

    #
    # end of method

    def resolve(self, l1, l2):
        """
        method: resolve

        return: bool

        description:
         this function takes 2 labels and resolves the
          equivalency.
        """

        r1 = self.rlabel[l1]
        r2 = self.rlabel[l2]
        if r1 < r2:
            self.plabel[r1].update(self.plabel[r2])
            for r in self.plabel[r1]:
                self.rlabel[r] = r1

        elif r2 < r1:
            self.plabel[r2].update(self.plabel[r1])
            for r in self.plabel[r2]:
                self.rlabel[r] = r2

        return True
    #
    # end of method

    def dtree9(self, voxel):
        """
        method: dtree9
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """
        t9 = (voxel[0] + self.dtree[9][0], voxel[1] + self.dtree[9][1], voxel[2] + self.dtree[9][2])
        return self.imgID[t9]

    #
    # end of method

    def dtree3(self, voxel):
        """
        method: dtree3
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """
        t3 = (voxel[0] + self.dtree[3][0], voxel[1] + self.dtree[3][1], voxel[2] + self.dtree[3][2])
        t1 = (voxel[0] + self.dtree[1][0], voxel[1] + self.dtree[1][1], voxel[2] + self.dtree[1][2])
        t8 = (voxel[0] + self.dtree[8][0], voxel[1] + self.dtree[8][1], voxel[2] + self.dtree[8][2])
        t10 = (voxel[0] + self.dtree[10][0], voxel[1] + self.dtree[10][1], voxel[2] + self.dtree[10][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        # check voxel 12, 11, 13
        #
        if self.imgID[t12] > 0:
            if self.imgID[t8] + self.imgID[t10] + self.imgID[t1] == 0:
                # resolve t3 t12
                #
                self.resolve(self.imgID[t3], self.imgID[t12])
        else:
            if (self.imgID[t11] > 0 and self.imgID[t8] + self.imgID[t1] == 0):
                # resolve t3 t11
                #
                self.resolve(self.imgID[t3], self.imgID[t11])

            if (self.imgID[t13] > 0 and self.imgID[t10] == 0):
                # resolve t3 t13
                #
                self.resolve(self.imgID[t3], self.imgID[t13])

        return self.imgID[t3]
    #
    # end of method

    def dtree6(self, voxel):
        """
        method: dtree6
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t6 = (voxel[0] + self.dtree[6][0], voxel[1] + self.dtree[6][1], voxel[2] + self.dtree[6][2])
        t1 = (voxel[0] + self.dtree[1][0], voxel[1] + self.dtree[1][1], voxel[2] + self.dtree[1][2])
        t8 = (voxel[0] + self.dtree[8][0], voxel[1] + self.dtree[8][1], voxel[2] + self.dtree[8][2])
        t10 = (voxel[0] + self.dtree[10][0], voxel[1] + self.dtree[10][1], voxel[2] + self.dtree[10][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        # set root to t6
        #

        # check voxel 12, 11, 13
        #
        if self.imgID[t12] > 0:
            if self.imgID[t8] + self.imgID[t10] + self.imgID[t1] == 0:
                # resolve t6 t12
                #
                self.resolve(self.imgID[t6], self.imgID[t12])

        else:
            if (self.imgID[t11] > 0 and self.imgID[t8] + self.imgID[t1] == 0):
                # resolve t6 t11
                #
                self.resolve(self.imgID[t6], self.imgID[t11])

            if (self.imgID[t13] > 0 and self.imgID[t10] == 0):
                # resolve t6 t13
                #
                self.resolve(self.imgID[t6], self.imgID[t13])

        return self.imgID[t6]
    #
    # end of method

    def dtree1(self, voxel):
        """
        method: dtree1
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t1 = (voxel[0] + self.dtree[1][0], voxel[1] + self.dtree[1][1], voxel[2] + self.dtree[1][2])
        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        t10 = (voxel[0] + self.dtree[10][0], voxel[1] + self.dtree[10][1], voxel[2] + self.dtree[10][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t10] > 0):
            if self.imgID[t12] == 0:
                # resolve t1 t10
                #
                self.resolve(self.imgID[t1], self.imgID[t10])

        else:
            if self.imgID[t4] > 0:
                # resolve t1 t4
                #
                self.resolve(self.imgID[t1], self.imgID[t4])

            else:
                if self.imgID[t7] > 0:
                    # resolve t1 t7
                    #
                    self.resolve(self.imgID[t1], self.imgID[t7])

            if self.imgID[t13] > 0:
                if self.imgID[t12] == 0:
                    # resolve t1 t13
                    #
                    self.resolve(self.imgID[t1], self.imgID[t13])

        return self.imgID[t1]
    #
    # end of method

    def dtree8(self, voxel):
        """
        method: dtree8
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t8 = (voxel[0] + self.dtree[8][0], voxel[1] + self.dtree[8][1], voxel[2] + self.dtree[8][2])
        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        t10 = (voxel[0] + self.dtree[10][0], voxel[1] + self.dtree[10][1], voxel[2] + self.dtree[10][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t10] > 0):
            if self.imgID[t12] == 0:
                # resolve t8 t10
                #
                self.resolve(self.imgID[t8], self.imgID[t10])

        else:
            if self.imgID[t4] > 0:
                # resolve t8 t4
                #
                self.resolve(self.imgID[t8], self.imgID[t4])

            else:
                if self.imgID[t7] > 0:
                    # resolve t8 t7
                    #
                    self.resolve(self.imgID[t8], self.imgID[t7])

            if self.imgID[t13] > 0:
                if self.imgID[t12] == 0:
                    # resolve t8 t13
                    #
                    self.resolve(self.imgID[t8], self.imgID[t13])

        return self.imgID[t8]
    #
    # end of method

    def dtree10(self, voxel):
        """
        method: dtree10
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t10 = (voxel[0] + self.dtree[10][0], voxel[1] + self.dtree[10][1], voxel[2] + self.dtree[10][2])
        t2 = (voxel[0] + self.dtree[2][0], voxel[1] + self.dtree[2][1], voxel[2] + self.dtree[2][2])
        t5 = (voxel[0] + self.dtree[5][0], voxel[1] + self.dtree[5][1], voxel[2] + self.dtree[5][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])

        if self.imgID[t11] > 0 and self.imgID[t12] == 0:
            # resolve t10 t11
            #
            self.resolve(self.imgID[t10], self.imgID[t11])

        if self.imgID[t2] > 0:
            # resolve t10 t2
            #
            self.resolve(self.imgID[t10], self.imgID[t2])
        else:
            if self.imgID[t5] > 0:
                # resolve t10 t5
                #
                self.resolve(self.imgID[t10], self.imgID[t5])

        return self.imgID[t10]
    #
    # end of method

    def dtree2(self, voxel):
        """
        method: dtree2
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t2 = (voxel[0] + self.dtree[2][0], voxel[1] + self.dtree[2][1], voxel[2] + self.dtree[2][2])
        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t12] > 0):
            # resolve t2 t12
            #
            self.resolve(self.imgID[t2], self.imgID[t12])

        else:
            if (self.imgID[t11] > 0):
                # resolve t2 t11
                #
                self.resolve(self.imgID[t2], self.imgID[t11])
            if (self.imgID[t13] > 0):
                # resolve t2 t13
                #
                self.resolve(self.imgID[t2], self.imgID[t13])
        if (self.imgID[t4] > 0):
            # resolve t2 t4
            #
            self.resolve(self.imgID[t2], self.imgID[t4])
        else:
            if (self.imgID[t7] > 0):
                # resolve t2 t7
                #
                self.resolve(self.imgID[t2], self.imgID[t7])

        return self.imgID[t2]
    #
    # end of method

    def dtree5(self, voxel):
        """
        method: dtree5
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t5 = (voxel[0] + self.dtree[5][0], voxel[1] + self.dtree[5][1], voxel[2] + self.dtree[5][2])
        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t12] > 0):
            # resolve t5 t12
            #
            self.resolve(self.imgID[t5], self.imgID[t12])

        else:
            if (self.imgID[t11] > 0):
                # resolve t5 t11
                #
                self.resolve(self.imgID[t5], self.imgID[t11])
            if (self.imgID[t13] > 0):
                # resolve t5 t13
                #
                self.resolve(self.imgID[t5], self.imgID[t13])
        if (self.imgID[t4] > 0):
            # resolve t5 t4
            #
            self.resolve(self.imgID[t5], self.imgID[t4])
        else:
            if (self.imgID[t7] > 0):
                # resolve t5 t7
                #
                self.resolve(self.imgID[t5], self.imgID[t7])

        return self.imgID[t5]
    #
    # end of method

    def dtree12(self, voxel):
        """
        method: dtree12
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t12 = (voxel[0] + self.dtree[12][0], voxel[1] + self.dtree[12][1], voxel[2] + self.dtree[12][2])
        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        if (self.imgID[t4] > 0):
            # resolve t12 t4
            #
            self.resolve(self.imgID[t12], self.imgID[t4])
        else:
            if (self.imgID[t7] > 0):
                # resolve t12 t7
                #
                self.resolve(self.imgID[t12], self.imgID[t7])

        return self.imgID[t12]
    #
    # end of method

    def dtree4(self, voxel):
        """
        method: dtree4
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t4 = (voxel[0] + self.dtree[4][0], voxel[1] + self.dtree[4][1], voxel[2] + self.dtree[4][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t11] > 0):
            # resolve t4 t11
            #
            self.resolve(self.imgID[t4], self.imgID[t11])
        if (self.imgID[t13] > 0):
            # resolve t4 t13
            #
            self.resolve(self.imgID[t4], self.imgID[t13])

        return self.imgID[t4]
    #
    # end of method

    def dtree7(self, voxel):
        """
        method: dtree7
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t7 = (voxel[0] + self.dtree[7][0], voxel[1] + self.dtree[7][1], voxel[2] + self.dtree[7][2])
        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t11] > 0):
            # resolve t7 t11
            #
            self.resolve(self.imgID[t7], self.imgID[t11])
        if (self.imgID[t13] > 0):
            # resolve t7 t13
            #
            self.resolve(self.imgID[t7], self.imgID[t13])

        return self.imgID[t7]
    #
    # end of method

    def dtree11(self, voxel):
        """
        method: dtree11
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t11 = (voxel[0] + self.dtree[11][0], voxel[1] + self.dtree[11][1], voxel[2] + self.dtree[11][2])
        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])

        if (self.imgID[t13] > 0):
            # resolve t11 t13
            #
            self.resolve(self.imgID[t11], self.imgID[t13])

        return self.imgID[t11]
    #
    # end of method

    def dtree13(self, voxel):
        """
        method: dtree13
        return: int
        description:
         this function finds the minimum anterior voxel
          based on the optimized checking order
        """

        t13 = (voxel[0] + self.dtree[13][0], voxel[1] + self.dtree[13][1], voxel[2] + self.dtree[13][2])
        # everthying else empty
        #

        return self.imgID[t13]
    #
    # end of method

    def  findMinTag(self, voxel, tag, channel):
        """
        method: findMinTag

        return:

        description:
         find minimum tag of 13 anterior voxels within a 1 voxel radius
        """
        x = voxel[0]
        y = voxel[1]
        z = voxel[2]

        clr = channel
        minTag = tag
        # check 3x3 above voxel
        #
        if z > 0:
            """

            sliceZ = np.s_[max(0,z-1): min(z+1,self.nSlices)]
            sliceY = np.s_[max(0,y-1): min(y+1,self.length-1)+1]
            sliceX = np.s_[max(0,x-1): min(x+1,self.width-1)+1]

            nonzero = self.img[sliceZ, sliceY, sliceX].nonzero()
            minANT = minTag
            try:
                minANT = (self.img[sliceZ, sliceY, sliceX][nonzero]).min()
            except:
                minANT = minTag

            minTag = min(minTag, minANT)

            """



            for antY in range(max(0, y - 1), min(y + 1, self.length - 1) + 1):
                for antX in range(max(0, x - 1), min(x + 1, self.width - 1) + 1):
                    if (pt := self.img[z - 1, antY, antX]) > 0:
                        minTag = min(pt, minTag)


        # check 1x3 above voxel
        #
        if y > 0:
            for antX in range(max(0, x - 1), min(x + 1, self.width - 1) + 1):
                if (pt := self.img[z, y - 1, antX]) > 0:
                    minTag = min(pt, minTag)

        # check previous voxel
        #  (x - 1, y, z)
        if x > 0:
            if (pt := self.img[z, y, x - 1]) > 0:
                minTag = min(pt, minTag)

        return minTag
    #
    # end of method

    def writeVoxelList(self, fp):
        """
        method: writeVoxelList

        arguments:
         fp: file pointer

        return: bool

        description:
         take file pointer and write voxel list
        """

        fp.write("%s version = %s" % (nft.DELIM_COMMENT ,nft.CSV_VERSION) + \
                                    nft.DELIM_NEWLINE)
        fp.write("%s width = %d, length = %d, height = %d" %
                 (nft.DELIM_COMMENT, self.width, self.length, self.nSlices) + \
                    nft.DELIM_NEWLINE)
        fp.write("%s, %s, %s, %s, %s" %
                 ("obj", "index", "x", "y", "z") + \
                    nft.DELIM_NEWLINE)
        for obj, array in self.vdict.items():
            i = 0
            for item in array:
                fp.write("%d, %d, %d, %d, %d" % (obj, i, item[0], item[1], item[2]) + \
                            nft.DELIM_NEWLINE)
                i += 1

    #
    # end of method

    def writeTiff(self, ofile):
        """
        method: writeTiff
        args: ofile
        return: bool
        description:
         this function writes ndarray to a tif file
        """

        tfl.imwrite(ofile, self.imgID[1:self.nSlices + 1, 1:self.length + 1, 1:self.width + 1], imagej=True, dtype='uint16')
        return True

    #
    # end of method



#
# end of file
