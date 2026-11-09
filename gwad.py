import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from gwad_sub import ExampleSave, Screener
from utilities.load_model import graphed_forward


_DS_TRANSFORM = T.Compose([
    #T.Grayscale(num_output_channels=1),
    T.Resize((32, 32))
])


USE_HX_CACHE = True
USE_CUDA_GRAPH = True


def _ds_feature(x):
    # per-example feature normalisation strategy (computed once per queued example)
    # 1. dimensionality normalisation
    #   - (discarded) convert to grayscale
    #   - convert to low dimension space using bilinear resizing method
    if tuple(x.shape[-2:]) != (32, 32):
        # (T.Resize to the size the image already has returns it unchanged)
        x = _DS_TRANSFORM(x)
    x = x.flatten().cpu()

    # 2. distribution normalisation
    #   - transpose to normal distribution with mu = 0.0
    #   - (discarded) standardise the length of vectors. unit vector
    x = x - torch.mean(x)
    return x


def _ds_from_features(x0, x1, x2, d0=None, d0_norm=None):
    # calculate delta feature (d0 / d0_norm may be passed in when already known)
    if d0 is None:
        d0 = (x1 - x0)
        d0_norm = torch.norm(d0)
    d1 = (x2 - x1)

    # delta similarity - cosine angle of two delta
    dot_product = torch.matmul(d0, d1)
    d1_norm = torch.norm(d1)
    if d0_norm == 0 or d1_norm == 0:
        raw_angle = torch.tensor(1e-32)
    else:
        raw_angle = dot_product / (d0_norm * d1_norm)
    abs_angle = torch.abs(raw_angle)

    if torch.isnan(abs_angle):
        print("warning : angle is nan!!")

    ds = raw_angle
    return ds, abs_angle, d0_norm, d1_norm, d1


def _hods_bin(ds, bin_size):
    # index of ds in a histogram over [-1, 1]: int(|(-1) - ds| // bin_size), evaluated in
    # float32 exactly as the original torch ops on the float32 ds tensor did
    return int(np.abs(np.float32(-1) - np.float32(float(ds))) // np.float32(bin_size))


def extract_ds(x0, x1, x2):
    return _ds_from_features(_ds_feature(x0), _ds_feature(x1), _ds_feature(x2))[:4]


class GWAD():
    def __init__(self, device, cfg, stats, mode='defence', model=None, delta_net=None):
        super().__init__()
        self.name = 'gwad_v5'
        self.device = device
        self.stats = stats
        self.mode = mode
        self.model = model
        self.delta_net = delta_net
        # batch-1 target-model forward replayed as a CUDA graph (bit-identical; eager fallback on CPU)
        self._model_fwd = graphed_forward(model) if model is not None else None
        self.example_save = ExampleSave(cfg["model"]["data_mu"], cfg["model"]["data_std"], cfg["model"]["data_format"])
        enable = cfg["screen"]["on"]
        thold = cfg["screen"]["thold"]
        fifo_depth = cfg["screen"]["n"]
        self.screener = Screener(enable=enable, thold=thold, sigma=3, n=fifo_depth, dim=32)

        # example queue to calculate ds, plus the cached ds feature (_ds_feature)
        # of each queued example, kept in lockstep with example_q
        self.example_q = []
        self.feat_q = []
        # (f1, f2, d1, d1_norm) of the last ds: d1 = f2 - f1 is the next d0 when the
        # queue has shifted by one example
        self._last_delta = None

        # ds configurations
        self.ds_size = 16*16
        self.ds_q = []
        self.ds_out = 0
        self.ds_in = 0
        # hods bin index of each queued ds (in lockstep with ds_q)
        self.bin_q = []
        self.bin_out = 0
        self.bin_in = 0

        # hods configuration
        self.hods_bins = 200
        self.hods_vector = [0]  # HoDS counts, float32 numpy array (1, hods_bins + 1)
        self.hods_flag = False
        # Delta-Net prediction for each HoDS seen (deterministic network, so the same
        # histogram always yields the same class); cleared on reset
        self._hx_cache = {}
        self._hx_cache_max = 1 << 16
        # captured Delta-Net CUDA graph: (key, graph, static input, static output)
        self._graph = None
        self._graph_failed = False

        self.show_detect = True
        self.query_cnt = 0
        self.delta_query_cnt = 0

        self.classes = delta_net.classes
        self.hx = torch.zeros(7)

        self.stats.classes = self.classes
        self.query_to_check = []

        self.suspicious = False

        # per-query record for visualisation (tools/visualize.py), enabled by --save_dir
        self.recorder = None
        self.last_pred = None
        self.first_detect = None

    def defended_model(self, x):
        self.query_cnt += 1
        return self._model_fwd(x)

    def delta_model(self, feat):
        # feat may live on the CPU; it is moved to the Delta-Net's device here
        self.delta_query_cnt += 1
        self.delta_net.eval()
        output = self._delta_forward(feat)
        _, hx = output.data.max(1)
        return hx

    def _delta_forward(self, feat):
        # Delta-Net forward. On CUDA the (fixed-shape, batch-1) forward is captured
        # once into a CUDA graph and replayed: the same kernels run on the same
        # weights, so the output is bit-identical to the eager forward while the
        # ~20 kernel launches per query become one graph launch.
        dev = torch.device(self.device)
        if USE_CUDA_GRAPH and dev.type == 'cuda' and not torch.is_grad_enabled():
            key = (id(self.delta_net), tuple(feat.shape), feat.dtype)
            g = self._graph if self._graph is not None and self._graph[0] == key else None
            if g is None and not self._graph_failed:
                try:
                    g = self._capture_delta_graph(key, feat, dev)
                except Exception as e:  # fall back to the eager forward
                    print('GWAD: Delta-Net CUDA graph capture failed (%s); using eager forward' % e)
                    self._graph_failed = True
                    g = None
            if g is not None:
                _, gr, static_in, static_out = g
                static_in.copy_(feat)
                gr.replay()
                return static_out
        return self.delta_net(feat.to(dev))

    def _capture_delta_graph(self, key, feat, dev):
        static_in = torch.zeros(feat.shape, dtype=feat.dtype, device=dev)
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                self.delta_net(static_in)
        torch.cuda.current_stream(dev).wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            static_out = self.delta_net(static_in)
        self._graph = (key, gr, static_in, static_out)
        return self._graph

    def reset(self):
        self.stats.reset()
        self.query_cnt = 0
        self.delta_query_cnt = 0
        self.hx = torch.zeros(7)
        self.show_detect = True
        self.screener.reset()
        self.example_q.clear()
        self.feat_q.clear()
        self._last_delta = None
        self.ds_q.clear()
        self.bin_q.clear()
        self.hods_flag = False
        self._hx_cache.clear()
        self.first_detect = None
        print("\nGWAD_%s starts" % (self.mode))
        return

    def examples_ready(self):
        # check if three examples are queued
        if len(self.example_q) == 3:
            return True
        return False

    @staticmethod
    def host_copy(x):
        # single device->host copy; the screener image and the ds feature are both
        # computed from it (elementwise float32 ops, identical on CPU and GPU)
        xc = x.detach().cpu()
        if xc is x:
            xc = x.clone()
        return xc

    def examples_queue(self, x, xc=None):
        # xc: host copy of x made by the caller (see run), otherwise made here
        if self.examples_ready():
            # if examples queue is full shift and empty oldest slot for new example
            self.example_q.pop(0)
            self.feat_q.pop(0)

        if xc is None:
            xc = self.host_copy(x)

        self.suspicious = True
        # check if example is suspicious
        self.suspicious = self.screener.run(self.example_save.ready_img(xc))
        if self.suspicious:
            # suspicious query, pass to gwad
            self.example_q.append(xc)
            self.feat_q.append(_ds_feature(xc))
        return

    def dss_ready(self):
        length = len(self.ds_q)
        if length == self.ds_size:
            return True
        elif length > self.ds_size:
            return True
        else:
            return False

    def ds_queue(self, ds):
        # if ds queue is full shift and empty oldest slot for new ds
        b = _hods_bin(ds, 2 / self.hods_bins)
        if self.dss_ready():
            self.ds_in = ds
            self.ds_out = self.ds_q.pop(0)
            self.bin_in = b
            self.bin_out = self.bin_q.pop(0)
        self.ds_q.append(ds)
        self.bin_q.append(b)
        return

    def set_hods_element(self, ele, action, b_idx=None):
        if b_idx is None:
            b_idx = _hods_bin(ele, 2 / self.hods_bins)
        if action == 'add':
            self.hods_vector[0, int(b_idx)] += 1
        elif action == 'remove':
            if self.hods_vector[0, int(b_idx)] > 0:
                self.hods_vector[0, int(b_idx)] -= 1
        else:
            raise ValueError('Invalid hods setting action')

    def make_hods(self):
        if self.dss_ready():
            feature = self.ds_q
            # generate historgram of delta similarity
            if self.hods_flag is False:
                # create hods feature vector
                self.hods_flag = True
                # (kept on the CPU; only the normalised feature goes to the Delta-Net)
                self.hods_vector = np.zeros((1, self.hods_bins + 1), dtype=np.float32)
                for i in range(len(feature)):
                    self.set_hods_element(feature[i], 'add', self.bin_q[i])
            else:
                # hods feature vector has been created, just modify hods with old and new ds.
                self.set_hods_element(self.ds_out, 'remove', self.bin_out)
                self.set_hods_element(self.ds_in, 'add', self.bin_in)
        return

    def attack_detect(self):
        flag = False
        if self.dss_ready():
            key = self.hods_vector.tobytes()
            hx = self._hx_cache.get(key) if USE_HX_CACHE else None
            if hx is None:
                # normalise hods (float32, same arithmetic as on the GPU)
                m0 = self.hods_vector.min()
                m1 = self.hods_vector.max()
                with np.errstate(divide='ignore', invalid='ignore'):
                    hods = (self.hods_vector - m0) / (m1 - m0)

                # Delta-Net predicts the type of attack
                hx = int(self.delta_model(torch.from_numpy(hods)))
                if len(self._hx_cache) >= self._hx_cache_max:
                    self._hx_cache.clear()
                self._hx_cache[key] = hx
            else:
                # same HoDS as an earlier query: Delta-Net output is the same
                self.delta_query_cnt += 1

            # check Delta-Net prediction
            self.last_pred = self.classes[hx]
            if self.classes[hx] == 'benign':
                # failed - benign
                flag = False
            else:
                flag = True
                if self.first_detect is None:
                    self.first_detect = self.query_cnt
                if self.show_detect:
                    print("GWAD_%s : first detection [%s] made @ %dth query" % (self.mode, self.classes[hx], self.query_cnt))
                    self.show_detect = False

            if self.mode == 'defence' or self.mode == 'simulate':
                self.hx[hx] += 1
        return flag

    def calculate_ds(self):
        # get the cached features of the three consecutive queued examples
        f0, f1, f2 = self.feat_q

        # reuse d0 = f1 - f0 when it was the previous query's d1
        last = self._last_delta
        if last is not None and last[0] is f0 and last[1] is f1:
            d0, d0_norm = last[2], last[3]
        else:
            d0, d0_norm = None, None

        # get delta similarity
        ds, _, d0_norm, d1_norm, d1 = _ds_from_features(f0, f1, f2, d0, d0_norm)
        self._last_delta = (f1, f2, d1, d1_norm)

        return ds, d0_norm

    def save_query(self, q):
        #self.example_save.sequence(q.x, self.query_cnt)
        if q.t == 'attack':
            # attack query is injected
            if self.suspicious:
                # correctly detected attack so pass to follows
                self.stats.attack_passed += 1
                #self.example_save.attack_passed(q.x, self.query_cnt)
            else:
                # error - attack is seen to be dummy
                self.stats.attack_screened += 1
                #self.example_save.attack_screened(q.x, self.query_cnt)
        elif q.t == 'benign':
            # dummy query is injected
            if self.suspicious:
                # error - dummy is seen to be attack
                self.stats.dummy_passed += 1
                #self.example_save.dummy_passed(q.x, self.query_cnt)
            else:
                # correctly detected dummy so screen out
                self.stats.dummy_screened += 1
                #self.example_save.dummy_screened(q.x, self.query_cnt)
        else:
            UserWarning("aa")
        return

    def run(self, q):
        # q.t - query type: 'dummy' or 'attack'
        # q.x - example

        # get example
        # (no copy needed: the model does not modify its input and the detector
        # keeps its own host copy of the example)
        x = q.x.detach()

        # copy the example to the host BEFORE launching the target model: the copy waits for all queued GPU
        # work, so taking it after the launch would make the CPU idle until the forward pass finishes.
        # this way the (asynchronous) forward pass runs on the GPU while the CPU does the detection below;
        # the GPU work and its order are unchanged, so results are identical
        xc = self.host_copy(x) if self.mode != 'off' else None

        if self.mode != 'simulate':
            # query to target model to defend
            output = self.defended_model(x)
        else:
            output = []

        # see enabled
        if self.mode == 'off':
            return output

        # -- attack detection routine --
        # pre-process and example queue
        self.examples_queue(x, xc)

        # save query and update pre-process stats
        if self.mode == 'defence':
            self.save_query(q)

        # check example queue is full for detection
        self.last_pred = None
        if self.examples_ready() is False:
            self.record_query(q, None)
            return output

        # calculate delta similarity
        ds, norm = self.calculate_ds()

        # update ds vector
        self.ds_queue(ds)

        # make and update hods feature vector
        self.make_hods()

        # update stats
        self.stats.update(ds, norm)

        # check under attack
        self.attack_detect()
        self.record_query(q, ds)
        return output

    def record_query(self, q, ds):
        if self.recorder is not None and self.mode == 'defence':
            self.recorder.log_query(self.query_cnt, q.t, not self.suspicious, ds, self.last_pred, q.x)

    def get_stats(self, image):
        x = image.clone().detach()
        update = self.check_update(x)
        return update

    def get_predictions(self):
        return self.hx

    def get_queries(self):
        return self.query_cnt, self.delta_query_cnt

    def show_predictions(self):
        print("GWAD_%s : prediction result - delta-net predicted total {} ".format(self.mode, torch.sum(self.hx)))
        for i in range(len(self.classes)):
            print("%s[%d], " % (self.classes[i], self.hx[i]), end='')
        print("")
        return
