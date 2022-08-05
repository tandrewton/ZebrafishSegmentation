import cellpose, time
from cellpose import models

# purpose of extCellposeTest.py is to define my own stitch3D_test function, replacing stitch3D. 
# unfortunately, stitch3D is used in CellposeModel._run_cp which is used in CellposeModel.eval, which ...
#  is used in Cellpose.eval and SizeModel.eval
# the solution for this is to write a copy of each of these functions, but replacing eval and _run_cp with eval_test and _run_cp_test
#   where the only difference other than the names is that stitch3D has been replaced with stitch3D_test

def stitch3D_test(masks, stitch_threshold=0.25): #would normally be in utils.py 
    """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
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

class Cellpose(cellpose.models.Cellpose):
    pass
    # override/add any functions here.
    # in models.py, we already have self.cp = CellposeModel(...)

    def eval_test(self, x, batch_size=8, channels=None, channel_axis=None, z_axis=None,
             invert=False, normalize=True, diameter=30., do_3D=False, anisotropy=None,
             net_avg=False, augment=False, tile=True, tile_overlap=0.1, resample=True, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0, min_size=15, stitch_threshold=0.0, 
             rescale=None, progress=None, model_loaded=False):
        """ run cellpose and get masks
        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)
        channels: list (optional, default None)
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].
        
        channel_axis: int (optional, default None)
            if None, channels dimension is attempted to be automatically determined
        z_axis: int (optional, default None)
            if None, z dimension is attempted to be automatically determined
        invert: bool (optional, default False)
            invert image pixel intensity before running network (if True, image is also normalized)
        normalize: bool (optional, default True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
        diameter: float (optional, default 30.)
            if set to None, then diameter is automatically estimated if size model is loaded
        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input
        anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)
        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False
        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment
        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended)
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows
        resample: bool (optional, default True)
            run dynamics at original image size (will be slower but create more accurate boundaries)
        interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)
        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)
        cellprob_threshold: float (optional, default 0.0)
            all pixels with value above threshold kept for masks, decrease to find more and larger masks
        min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1
        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D and equal image sizes, masks are stitched in 3D to return volume segmentation
        rescale: float (optional, default None)
            if diameter is set to None, and rescale is not None, then rescale is used instead of diameter for resizing image
        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI
        model_loaded: bool (optional, default False)
            internal variable for determining if model has been loaded, used in __main__.py
        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels
        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = XY flows at each pixel
            flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
            flows[k][3] = final pixel locations after Euler integration 
        styles: list of 1D arrays of length 256, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image
        diams: list of diameters, or float (if do_3D=True)
        """        

        tic0 = time.time()
        channels = [0,0] if channels is None else channels # why not just make this a default in the function header?

        estimate_size = True if (diameter is None or diameter==0) else False
        
        if estimate_size and self.pretrained_size is not None and not do_3D and x[0].ndim < 4:
            tic = time.time()
            models_logger.info('~~~ ESTIMATING CELL DIAMETER(S) ~~~')
            diams, _ = self.sz.eval_test(x, channels=channels, channel_axis=channel_axis, invert=invert, batch_size=batch_size, 
                                    augment=augment, tile=tile, normalize=normalize)
            rescale = self.diam_mean / np.array(diams)
            diameter = None
            models_logger.info('estimated cell diameter(s) in %0.2f sec'%(time.time()-tic))
            models_logger.info('>>> diameter(s) = ')
            if isinstance(diams, list) or isinstance(diams, np.ndarray):
                diam_string = '[' + ''.join(['%0.2f, '%d for d in diams]) + ']'
            else:
                diam_string = '[ %0.2f ]'%diams
            models_logger.info(diam_string)
        elif estimate_size:
            if self.pretrained_size is None:
                reason = 'no pretrained size model specified in model Cellpose'
            else:
                reason = 'does not work on non-2D images'
            models_logger.warning(f'could not estimate diameter, {reason}')
            diams = self.diam_mean 
        else:
            diams = diameter

        tic = time.time()
        models.models_logger.info('~~~ FINDING MASKS ~~~')
        masks, flows, styles = self.cp.eval_test(x, 
                                            batch_size=batch_size, 
                                            invert=invert, 
                                            normalize=normalize,
                                            diameter=diameter,
                                            rescale=rescale, 
                                            anisotropy=anisotropy, 
                                            channels=channels,
                                            channel_axis=channel_axis, 
                                            z_axis=z_axis,
                                            augment=augment, 
                                            tile=tile, 
                                            do_3D=do_3D, 
                                            net_avg=net_avg, 
                                            progress=progress,
                                            tile_overlap=tile_overlap,
                                            resample=resample,
                                            interp=interp,
                                            flow_threshold=flow_threshold, 
                                            cellprob_threshold=cellprob_threshold,
                                            min_size=min_size, 
                                            stitch_threshold=stitch_threshold,
                                            model_loaded=model_loaded)
        models_logger.info('>>>> TOTAL TIME %0.2f sec'%(time.time()-tic0))
    
        return masks, flows, styles, diams

class CellposeModel(cellpose.models.CellposeModel): 
    # override/add any functions here.

    # normally would be in models.py in CellposeModel class
    def eval_test(self, x, batch_size=8, channels=None, channel_axis=None, 
                z_axis=None, normalize=True, invert=False, 
                rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=False, 
                augment=False, tile=True, tile_overlap=0.1,
                resample=True, interp=True,
                flow_threshold=0.4, cellprob_threshold=0.0,
                compute_masks=True, min_size=15, stitch_threshold=0.0, progress=None,  
                loop_run=False, model_loaded=False):
        """
        segment list of images x, or 4D array - Z x nchan x Y x X
        Parameters
        ----------
        x: list or array of images
            can be list of 2D/3D/4D images, or array of 2D/3D/4D images
        batch_size: int (optional, default 8)
            number of 224x224 patches to run simultaneously on the GPU
            (can make smaller or bigger depending on GPU memory usage)
        channels: list (optional, default None)
            list of channels, either of length 2 or of length number of images by 2.
            First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
            Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
            For instance, to segment grayscale images, input [0,0]. To segment images with cells
            in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
            image with cells in green and nuclei in blue, input [[0,0], [2,3]].
        channel_axis: int (optional, default None)
            if None, channels dimension is attempted to be automatically determined
        z_axis: int (optional, default None)
            if None, z dimension is attempted to be automatically determined
        normalize: bool (default, True)
            normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
        invert: bool (optional, default False)
            invert image pixel intensity before running network
        diameter: float (optional, default None)
            diameter for each image, 
            if diameter is None, set to diam_mean or diam_train if available
        rescale: float (optional, default None)
            resize factor for each image, if None, set to 1.0;
            (only used if diameter is None)
        do_3D: bool (optional, default False)
            set to True to run 3D segmentation on 4D image input
        anisotropy: float (optional, default None)
            for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)
        net_avg: bool (optional, default False)
            runs the 4 built-in networks and averages them if True, runs one network if False
        augment: bool (optional, default False)
            tiles image with overlapping tiles and flips overlapped regions to augment
        tile: bool (optional, default True)
            tiles image to ensure GPU/CPU memory usage limited (recommended)
        tile_overlap: float (optional, default 0.1)
            fraction of overlap of tiles when computing flows
        resample: bool (optional, default True)
            run dynamics at original image size (will be slower but create more accurate boundaries)
        interp: bool (optional, default True)
            interpolate during 2D dynamics (not available in 3D) 
            (in previous versions it was False)
        flow_threshold: float (optional, default 0.4)
            flow error threshold (all cells with errors below threshold are kept) (not used for 3D)
        cellprob_threshold: float (optional, default 0.0) 
            all pixels with value above threshold kept for masks, decrease to find more and larger masks
        compute_masks: bool (optional, default True)
            Whether or not to compute dynamics and return masks.
            This is set to False when retrieving the styles for the size model.
        min_size: int (optional, default 15)
            minimum number of pixels per mask, can turn off with -1
        stitch_threshold: float (optional, default 0.0)
            if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation
        progress: pyqt progress bar (optional, default None)
            to return progress bar status to GUI
                        
        loop_run: bool (optional, default False)
            internal variable for determining if model has been loaded, stops model loading in loop over images
        model_loaded: bool (optional, default False)
            internal variable for determining if model has been loaded, used in __main__.py
        Returns
        -------
        masks: list of 2D arrays, or single 3D array (if do_3D=True)
            labelled image, where 0=no masks; 1,2,...=mask labels
        flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
            flows[k][0] = XY flow in HSV 0-255
            flows[k][1] = XY flows at each pixel
            flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
            flows[k][3] = final pixel locations after Euler integration 
        styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
            style vector summarizing each image, also used to estimate size of objects in image
        """
        
        if isinstance(x, list) or x.squeeze().ndim==5:
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            for i in iterator:
                maski, flowi, stylei = self.eval_test(x[i], 
                                                batch_size=batch_size, 
                                                channels=channels[i] if (len(channels)==len(x) and 
                                                                        (isinstance(channels[i], list) or isinstance(channels[i], np.ndarray)) and
                                                                        len(channels[i])==2) else channels, 
                                                channel_axis=channel_axis, 
                                                z_axis=z_axis, 
                                                normalize=normalize, 
                                                invert=invert, 
                                                rescale=rescale[i] if isinstance(rescale, list) or isinstance(rescale, np.ndarray) else rescale,
                                                diameter=diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter, 
                                                do_3D=do_3D, 
                                                anisotropy=anisotropy, 
                                                net_avg=net_avg, 
                                                augment=augment, 
                                                tile=tile, 
                                                tile_overlap=tile_overlap,
                                                resample=resample, 
                                                interp=interp,
                                                flow_threshold=flow_threshold,
                                                cellprob_threshold=cellprob_threshold, 
                                                compute_masks=compute_masks, 
                                                min_size=min_size, 
                                                stitch_threshold=stitch_threshold, 
                                                progress=progress,
                                                loop_run=(i>0),
                                                model_loaded=model_loaded)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
            return masks, flows, styles
        
        else:
            if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):
                self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
                
            # reshape image (normalization happens in _run_cp)
            x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                        do_3D=(do_3D or stitch_threshold>0), 
                                        normalize=False, invert=False, nchan=self.nchan)
            if x.ndim < 4:
                x = x[np.newaxis,...]
            self.batch_size = batch_size

            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                diameter = self.diam_labels
                rescale = self.diam_mean / diameter

            masks, styles, dP, cellprob, p = self._run_cp_test(x, 
                                                        compute_masks=compute_masks,
                                                        normalize=normalize,
                                                        invert=invert,
                                                        rescale=rescale, 
                                                        net_avg=net_avg, 
                                                        resample=resample,
                                                        augment=augment, 
                                                        tile=tile, 
                                                        tile_overlap=tile_overlap,
                                                        flow_threshold=flow_threshold,
                                                        cellprob_threshold=cellprob_threshold, 
                                                        interp=interp,
                                                        min_size=min_size, 
                                                        do_3D=do_3D, 
                                                        anisotropy=anisotropy,
                                                        stitch_threshold=stitch_threshold,
                                                        )
            
            flows = [plot.dx_to_circ(dP), dP, cellprob, p]
            return masks, flows, styles
    
    #normally would be in models.py in CellposeModel module
    def _run_cp_test(self, x, compute_masks=True, normalize=True, invert=False, 
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
                    masks = stitch3D_test(masks, stitch_threshold=stitch_threshold)
            
            flow_time = time.time() - tic
            if nimg > 1:
                models_logger.info('masks created in %2.2fs'%(flow_time))
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(), p.squeeze()
            
        else:
            masks, p = np.zeros(0), np.zeros(0)  #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p

class SizeModel(cellpose.models.SizeModel):
    # override/add any functions here.
    def eval_test(self, x, channels=None, channel_axis=None, 
             normalize=True, invert=False, augment=False, tile=True,
             batch_size=8, progress=None, interp=True):
        """ use images x to produce style or use style input to predict size of objects in image
            Object size estimation is done in two steps:
            1. use a linear regression model to predict size from style in image
            2. resize image to predicted size and run CellposeModel to get output masks.
                Take the median object size of the predicted masks as the final predicted size.
            Parameters
            -------------------
            x: list or array of images
                can be list of 2D/3D images, or array of 2D/3D images
            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined
            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel
            invert: bool (optional, default False)
                invert image pixel intensity before running network
            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment
            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)
            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI
            Returns
            -------
            diam: array, float
                final estimated diameters from images x or styles style after running both steps
            diam_style: array, float
                estimated diameters from style alone
        """
        
        if isinstance(x, list):
            diams, diams_style = [], []
            nimg = len(x)
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            for i in iterator:
                diam, diam_style = self.eval_test(x[i], 
                                             channels=channels[i] if (channels is not None and len(channels)==len(x) and 
                                                                     (isinstance(channels[i], list) or isinstance(channels[i], np.ndarray)) and
                                                                     len(channels[i])==2) else channels,
                                             channel_axis=channel_axis, 
                                             normalize=normalize, 
                                             invert=invert,
                                             augment=augment,
                                             tile=tile,
                                             batch_size=batch_size,
                                             progress=progress,
                                            )
                diams.append(diam)
                diams_style.append(diam_style)

            return diams, diams_style

        if x.squeeze().ndim > 3:
            models_logger.warning('image is not 2D cannot compute diameter')
            return self.diam_mean, self.diam_mean

        styles = self.cp.eval_test(x, 
                              channels=channels, 
                              channel_axis=channel_axis, 
                              normalize=normalize, 
                              invert=invert, 
                              augment=augment, 
                              tile=tile,
                              batch_size=batch_size, 
                              net_avg=False,
                              resample=False,
                              compute_masks=False)[-1]

        diam_style = self._size_estimation(np.array(styles))
        diam_style = self.diam_mean if (diam_style==0 or np.isnan(diam_style)) else diam_style
        
        masks = self.cp.eval_test(x, 
                             compute_masks=True,
                             channels=channels, 
                             channel_axis=channel_axis, 
                             normalize=normalize, 
                             invert=invert, 
                             augment=augment, 
                             tile=tile,
                             batch_size=batch_size, 
                             net_avg=False,
                             resample=False,
                             rescale =  self.diam_mean / diam_style if self.diam_mean>0 else 1, 
                             #flow_threshold=0,
                             diameter=None,
                             interp=False,
                            )[0]
        
        diam = utils.diameters(masks)[0]
        diam = self.diam_mean if (diam==0 or np.isnan(diam)) else diam
        return diam, diam_style


class UnetModel(cellpose.core.UnetModel):
    pass 
    # override/add any functions here.

class TqdmToLogger(cellpose.utils.TqdmToLogger):
    pass 
    # override/add any functions here.