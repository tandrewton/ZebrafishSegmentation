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
    
    # override/add any functions here.
    def writeSomething(self):
        print("testing print function inside of inherited class!\n")
        self.cp.writeSomething()

class CellposeModel(cellpose.models.CellposeModel): 
    pass
    # override/add any functions here.

    def writeSomething(self):
        print("testing print function inside of inherited called class CellposeModel in extCellpose!\n")

    def stitch3D(self, masks, stitch_threshold=0.25): #would normally be in utils.py 
        """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
        print("currently in newly defined stitch3D!\n")
        mmax = masks[0].max()
        empty = 0
        
        for i in range(len(masks)-1):
            iou = metrics._intersection_over_union(masks[i+1], masks[i])[1:,1:]
            if not iou.size and empty == 0:
                masks[i+1] = masks[i+1]
                mmax = masks[i+1].max()
            elif not iou.size and not empty == 0:
                icount = masks[i+1].max()
                istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
                mmax += icount
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
            else:
                iou[iou < stitch_threshold] = 0.0
                iou[iou < iou.max(axis=0)] = 0.0
                istitch = iou.argmax(axis=1) + 1
                ino = np.nonzero(iou.max(axis=1)==0.0)[0]
                istitch[ino] = np.arange(mmax+1, mmax+len(ino)+1, 1, int)
                mmax += len(ino)
                istitch = np.append(np.array(0), istitch)
                masks[i+1] = istitch[masks[i+1]]
                empty = 1
                
        return masks

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, 
                flow_threshold=0.4, min_size=15,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
                ):
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
                
                if stitch_threshold > 0 and nimg > 1:
                    models_logger.info(f'stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks')
                    masks = self.stitch3D(masks, stitch_threshold=stitch_threshold)
            
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