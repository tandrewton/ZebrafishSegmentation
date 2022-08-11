from cellpose import * #import all, can result in unwanted namespace conflicts;
# be careful not to override functions unintentionally. use different function names when possible
import cellpose

import numpy as np
from tqdm import trange
import time, logging
from cellpose.models import models_logger

class extCellpose(cellpose.models.Cellpose):
    pass
    def __init__(self, gpu=False, model_type='cyto', net_avg=False, device=None):
        from cellpose.core import assign_device
        from cellpose.models import size_model_path
        super().__init__()
        self.torch = True
        
        # assign device (GPU or CPU)
        sdevice, gpu = assign_device(self.torch, gpu)
        self.device = device if device is not None else sdevice
        self.gpu = gpu
        
        model_type = 'cyto' if model_type is None else model_type
        
        self.diam_mean = 30. #default for any cyto model 
        nuclear = 'nuclei' in model_type
        if nuclear:
            self.diam_mean = 17. 
        
        self.cp = CellposeModel(device=self.device, gpu=self.gpu,
                                model_type=model_type,
                                diam_mean=self.diam_mean,
                                net_avg=net_avg)
        self.cp.model_type = model_type
        
        # size model not used for bacterial model
        self.pretrained_size = size_model_path(model_type, self.torch)
        self.sz = SizeModel(device=self.device, pretrained_size=self.pretrained_size,
                            cp_model=self.cp)
        self.sz.model_type = model_type

        # nuclear mask data, imported using readNuclearMaskFile
        self.nucMaskData = np.zeros(1)
    
    # override/add any functions here.
    def writeSomething(self):
        print("testing print function inside of inherited class!\n")
        self.cp.writeSomething()

    def readNuclearMaskFile(self, filename):
        self.nucMaskData = np.load(filename)

class CellposeModel(cellpose.models.CellposeModel): 
    pass
    # override/add any functions here.

    def writeSomething(self):
        print("testing print function inside of inherited called class CellposeModel in extCellpose!\n")

    def readMaskFile(self, filename): # read in an npy file and return it
        return np.load(filename)

    def calcNumOfNucPerCell(self, nucMask, cellMask):
        # compute number of nuclei per cell. Can take 2D or 3D masks as input
        # input: nd array of ints (0,1,2,...), nd array of ints (0,1,2,...) with same size (zSlices x pixelsX x pixelsY)
        # output: np array of ints (0,1,2,...) size (1 x #cell masks)
        pixelContactMap = metrics._label_overlap(nucMask, cellMask)
        overlaps = pixelContactMap/pixelContactMap
        overlaps[np.isnan(overlaps)] = 0
        print("shape of overlaps=", np.shape(overlaps))
        numNucsInCell = np.sum(overlaps,axis=0)
        numNucsInCell[0] = 0
        print("shape of numNucsInCell=", np.shape(numNucsInCell))
        return numNucsInCell

    def calcCentroidOfEachCell(self, mask):
        # from a mask, calculate the centroid of each unique mask value (each cell ID)
        # in 3D case, only compute the centroid for the first z-slice containing each cell ID

        # output: centroids, a np array of size (numMaskIDs x 3)
        #       and centroidZLocations, a np array of size (numMaskIDs)

        if (len(np.shape(mask)) == 2): # we have a 2D mask. Compute centroids, should be (numMaskIDs x 2)
            centroids = np.zeros([len(np.unique(mask2D)),3])
            centroidZLocations = np.array([]) # don't save any meaningful information in 2D case
            for i in np.unique(mask2D):
                # compute center of mass of elements in mask with same value as i
                x_centroid, y_centroid = np.argwhere(mask2D == i).mean(0)
                centroids[i] = [x_centroid, y_centroid]
            return centroids, centroidZLocations
        elif (len(np.shape(mask)) == 3): # we have a 3D mask. Compute centroid of first z-slice in each mask,
            #                                    and corresponding z-val for centroid, should be (numMaskIDs x 3)
            #                                       i.e. centroid[maskID] = (xloc, yloc, zloc)
            centroids = np.zeros([len(np.unique(mask3D)),3])
            centroidZLocations = np.zeros([len(np.unique(mask3D)),3])
            for i in np.unique(mask3D): #takes a few seconds for numPlanes = 20, but maybe not a bottleneck
                z = 0
                while (centroids[i,2] == 0 and z < np.shape(mask3D)[0]):
                    if (np.sum(mask3D[z] == i) > 0):
                        x_centroid, y_centroid = np.argwhere(mask3D[z] == i).mean(0)
                        centroids[i] = [x_centroid, y_centroid, z]
                        centroidZLocations[i] = z
                        break
                    else:
                        z+= 1
            return centroids, centroidZLocations

    def stitch3D(self, masks, stitch_threshold=0.25): #would normally be in utils.py. Cannot modify the arglist because of inheritance
        """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
        print("currently in newly defined stitch3D!\n")
        mmax = masks[0].max() # max of first frame
        empty = 0 # empty flips to 1 when iou.size > 0 
        
        for i in range(len(masks)-1): # z stack (frame) iterator
            iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            #numNucsInCell = calcNumOfNucPerCell()
            if not iou.size and empty == 0: # iou is empty
                masks[i+1] = masks[i+1]
                mmax = masks[i+1].max() # max of next frame
                print("mmax shape in first if clause: ", np.shape(mmax))
            elif not iou.size and not empty == 0: # iou is empty 
                icount = masks[i+1].max()
                istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
                mmax += icount # add max of next frame
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
                print("mmax shape in elif clause: ", np.shape(mmax), "icount=",icount,"istitch shape =",np.shape(istitch))
            else: # iou is nonempty
                iou[iou < stitch_threshold] = 0.0
                iou[iou < iou.max(axis=0)] = 0.0 # iou over axis 0 is the overlap of a new mask with each of the old masks
                istitch = iou.argmax(axis=1) + 1 # istitch is argmax of the columns of iou 
                print("before modification, istitch=", istitch)
                ino = np.nonzero(iou.max(axis=1)==0.0)[0] # ino = indices whenever iou's max along rows is zero
                # ino represents the positions where a lineage in the old masks has no parent in the new masks
                istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int) # arange gives an increasing range of cellIDs to accomodate new lineages
                # istitch[ino] replaces all of the dead lineages (ino) with new ones (arange)
                print("np.arange = ", np.arange(mmax+1, mmax+len(ino)+1, 1, int))
                mmax += len(ino)
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
                empty = 1
                print("iou shape = (new masks cells, old mask cells) = ", np.shape(iou))
                print("in else clause: istitch shape = ", np.shape(istitch), "ino shape = ", np.shape(ino))
                print("ino=", ino)
                print("after modification, istitch=", istitch)

        print("shape of stitch3D masks = ", np.shape(masks))
        return masks

    def stitch3DFlexibleThreshold(self, masks, nucMasks, numNucsPerCell, centroid, centroidZLocations, stitch_threshold=0.25):
        ''' To be called after running stitch3D for a second opinion on the stitching,
        evaluation is based on a flexible IoU threshold depending on whether stitch3D produced 
        cells with too many or too few nuclei'''
        print("currently in stitch3DFlexibleThreshold!\n")

        mmax = masks[0].max() # max of first frame
        empty = 0 # empty flips to 1 when iou.size > 0 
        stitch_threshold_low = stitch_threshold / 2.0
        stittch_threshold_high = stitch_threshold * 2.0
        indOfLoweredThresholds = np.array([])
        indOfRaisedThresholds = np.array([])
        
        for i in range(len(masks)-1): # z stack (frame) iterator
            iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            # stitch_threshold_mat = iou
            # stitch_threshold_mat[indOfLoweredThresholds] = stitch_threshold_low
            # stitch_threshold_mat[indOfRaisedThresholds] = stitch_threshold_high 
            # indOfRaisedThresholds = np.full(np.shape(stitch_threshold_mat), False) # reset indices of changed thresholds
            # indOfLoweredThresholds = indOfRaisedThresholds
            if not iou.size and empty == 0: # iou is empty
                masks[i+1] = masks[i+1]
                mmax = masks[i+1].max() # max of next frame
                print("mmax shape in first if clause: ", np.shape(mmax))
            elif not iou.size and not empty == 0: # iou is empty 
                icount = masks[i+1].max()
                istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
                mmax += icount # add max of next frame
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
                print("mmax shape in elif clause: ", np.shape(mmax), "icount=",icount,"istitch shape =",np.shape(istitch))
            else: # iou is nonempty
                iou[iou < stitch_threshold_mat] = 0.0
                iou[iou < iou.max(axis=0)] = 0.0 # iou over axis 0 is the overlap of a new mask with each of the old masks
                istitch = iou.argmax(axis=1) + 1 # istitch is argmax of the columns of iou 
                #print("before modification, istitch=", istitch)
                ino = np.nonzero(iou.max(axis=1)==0.0)[0] # ino = indices whenever iou's max along rows is zero
                # ino represents the positions where a lineage in the old masks has no parent in the new masks
                istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int) # arange gives an increasing range of cellIDs to accomodate new lineages
                # istitch[ino] replaces all of the dead lineages (ino) with new ones (arange)
                #print("np.arange = ", np.arange(mmax+1, mmax+len(ino)+1, 1, int))
                mmax += len(ino)
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
                empty = 1
                #print("iou shape = (new masks cells, old mask cells) = ", np.shape(iou))
                #print("in else clause: istitch shape = ", np.shape(istitch), "ino shape = ", np.shape(ino))
                #print("ino=", ino)
                #print("after modification, istitch=", istitch)

                centroidNew, centroidNewZLocations = calcCentroidOfEachCell(masks[i+1])
                print("shape centroidNew, ", np.shape(centroidNew))
                print("shape indThresholds, ", np.shape(indOfRaisedThresholds)) # hope that indOfRaisedThresholds is same size or larger than centroidNew
                if (i in centroidZLocations): # if frame i is known to have the first centroid of any cellIDs in old 3D mask
                    for j in range(0, np.shape(centroidNew)[0]): # let j iterate through list of centroids in frame i
                        for k in range(0, np.shape(centroid)[0]): # let k iterate through list of centroids in 3D mask
                        if (centroidNew[j,0] == centroid[k,0] and centroidNew[j,1] == centroid[k,1]): # match x,y in centroidNew to centroid
                            #if there's a match, then we check numNucs
                            if (numNucsPerCell[k] > 1):
                                indOfRaisedThresholds[j,:] = True
                                indOfRaisedThresholds[:,j] = True
                                # will probably have to just save the ID corresponding to k rather than what's currently written
                                # then, I'll check all the saved IDs against the current iteration of stitch_threshold_mat to determine what to overwrite
                            elif (numNucsPerCell[k] == 0):
                                indOfLoweredThresholds[j,:] = True
                                indOfLoweredThresholds[:,j] = True
                            
                # now I've set the thresholds differently if the new mask has first time masks in the 3D mask
                # however, I overwrite the thresholds every step. now I need to keep these thresholds lowered until this lineage dies

                # plan: the first time a lineage appears, cross-check it with the list of centroids. 
                #       if centroid(new lineage, z) == any of centroids[:,i]
                #           then if numNucsPerCell[centroid[_]] > 0
                #               lower its stitch_threshold 
        print("shape of stitch3D masks = ", np.shape(masks))
        return masks


    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, 
                flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
                ):
        print("inside _run_cp in inherited cp")
        tic = time.time()
        shape = x.shape
        nimg = shape[0]        
        
        bd, tr = None, None
        if do_3D:
            img = np.asarray(x)
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert)
            yf, styles = self._run_3D(img, rsz=rescale, anisotropy=anisotropy, 
                                      net_avg=net_avg, augment=augment, tile=tile,
                                      tile_overlap=tile_overlap)
            cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1] 
            dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                          axis=0) # (dZ, dY, dX)
            del yf
        else:
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            styles = np.zeros((nimg, self.nbase[-1]), np.float32)
            if resample:
                dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
                cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
                
            else:
                dP = np.zeros((2, nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
                cellprob = np.zeros((nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
                
            for i in iterator:
                img = np.asarray(x[i])
                if normalize or invert:
                    img = transforms.normalize_img(img, invert=invert)
                if rescale != 1.0:
                    img = transforms.resize_image(img, rsz=rescale)
                yf, style = self._run_nets(img, net_avg=net_avg,
                                           augment=augment, tile=tile,
                                           tile_overlap=tile_overlap)
                if resample:
                    yf = transforms.resize_image(yf, shape[1], shape[2])

                cellprob[i] = yf[:,:,2]
                dP[:, i] = yf[:,:,:2].transpose((2,0,1)) 
                if self.nclasses == 4:
                    if i==0:
                        bd = np.zeros_like(cellprob)
                    bd[i] = yf[:,:,3]
                styles[i] = style
            del yf, style
        styles = styles.squeeze()
        
        
        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info('network run in %2.2fs'%(net_time))

        if compute_masks:
            tic=time.time()
            niter = 200 if (do_3D and not resample) else (1 / rescale * 200)
            if do_3D:
                masks, p = dynamics.compute_masks(dP, cellprob, niter=niter, 
                                                      cellprob_threshold=cellprob_threshold,
                                                      flow_threshold=flow_threshold,
                                                      interp=interp, do_3D=do_3D, min_size=min_size,
                                                      resize=None,
                                                      use_gpu=self.gpu, device=self.device
                                                    )
            else:
                masks, p = [], []
                resize = [shape[1], shape[2]] if not resample else None
                for i in iterator:
                    outputs = dynamics.compute_masks(dP[:,i], cellprob[i], niter=niter, cellprob_threshold=cellprob_threshold,
                                                         flow_threshold=flow_threshold, interp=interp, resize=resize,
                                                         use_gpu=self.gpu, device=self.device)
                    masks.append(outputs[0])
                    p.append(outputs[1])
                    
                masks = np.array(masks)
                p = np.array(p)
                
                if stitch_threshold > 0 and nimg > 1: # prerequisite: existing nucMasks data
                    # first pass stitches masks together based on flat stitch_threshold
                    models_logger.info(f'stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks')
                    masks = self.stitch3D(masks, stitch_threshold=stitch_threshold) 
                    
                    # read in pre-calculated nuclear masks file using cellpose 3D 
                    nucMasks = self.readMaskFile('output_nuc_masks.npy')
                    
                    # calculate number of nuclei per cell mask in 3D
                    numNucsPerCell = self.calcNumOfNucPerCell(nucMasks, masks)
                    centroid, centroidZLocations = self.calcCentroidOfEachCell(masks) 
                    # centroid[i,:] = [xi,yi,zi] is a map between maskID i and the centroid of its first slice in z

                    # find first slice of each maskID, store the sliceID and centroid
                    #       i.e. for z z-slices, n maskIDs, save a (z x n x 2) matrix of (z, x, y) = centroid position in first slice of a mask 
                    masks = self.stitch3DFlexibleThreshold(masks, numNucsPerCell)

                    #todo: write stitch3DFlexibleThreshold: like stitch3D 
                    #   difference: when starting a new lineage, calculate the centroid.
                    #       if that centroid is tied to a maskID in the old mask with >1 nuclei per cell
                    #       then be more strict in stitch_threshold for that new lineage
            
            flow_time = time.time() - tic
            if nimg > 1:
                models_logger.info('masks created in %2.2fs'%(flow_time))
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(), p.squeeze()
            
        else:
            masks, p = np.zeros(0), np.zeros(0)  #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p

class SizeModel(cellpose.models.SizeModel):
    pass
    # override/add any functions here.

class UnetModel(cellpose.core.UnetModel):
    pass 
    # override/add any functions here.

class TqdmToLogger(cellpose.utils.TqdmToLogger):
    pass 
    # override/add any functions here.