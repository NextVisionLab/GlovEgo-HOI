from _pycocotools.cocoeval import COCOeval
import numpy as np

class CustomHandGloveCOCOeval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt by score
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > maxDet:
            dt = dt[0:maxDet]

        # load computed ious
        if len(gt) > 0 and len(dt) > 0:
            ious = self.ious[imgId, catId][0:len(gt), 0:len(dt)]
        else:
            ious = []

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m)
                    iou = min([t, 1-1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, continue
                        if gtm[tind, gind] > 0:
                            continue
                        # if dt matched to ignored gt, ignore it
                        if m > -1 and gtIg[m] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[gind, dind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[gind, dind]
                        m = gind
                        
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    
                    # --- LOGICA CUSTOM ---
                    # Controlla che l'attributo 'gloves' corrisponda
                    g = gt[m]
                    if g.get('gloves', -1) != d.get('gloves', -1):
                        continue # Se non corrispondono, non è un match valido
                    # ---------------------
                    
                    # ignore gt?
                    dtIg[tind, dind] = gtIg[m]
                    dtm[tind, dind] = gt[m]['id']
                    gtm[tind, m] = d['id']
                    
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        return {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }