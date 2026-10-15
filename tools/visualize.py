import os
import csv
import json
import time
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from gwad_sub import ExampleSave


class RunRecorder:
    """
    Save attack examples and GWAD decisions of one run (enabled by --save_dir).

    <save_dir>/<timestamp>_<run_name>/
        run_config.json                 arguments and json config of this run
        summary.csv                     one row per attacked image
        summary/img_XXXX.png            original | adversarial | perturbation (XXXX = test set index)
        gwad_timeline.csv               one row per query of every image
        gwad_timeline.png               detection rate over queries + first detection distribution
        queries/img_XXXX_qXXXXX.png     every `save_every` attack queries
    """
    def __init__(self, save_dir, run_name, run_config, cfg_model, data_classes, delta_classes, save_every=500):
        self.name = run_name
        self.dir = os.path.join(save_dir, '%s_%s' % (time.strftime('%Y%m%d-%H%M%S'), run_name))
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'run_config.json'), 'w') as f:
            json.dump(run_config, f, indent=2)

        self.example_save = ExampleSave(cfg_model["data_mu"], cfg_model["data_std"], 'numpy')
        self.data_classes = data_classes
        self.delta_classes = delta_classes
        self.save_every = save_every

        self.timeline_file = open(os.path.join(self.dir, 'gwad_timeline.csv'), 'w', newline='')
        self.timeline = csv.writer(self.timeline_file)
        self.timeline.writerow(['img', 'query', 'type', 'screened', 'ds', 'pred'])

        self.summary_rows = []
        self.detect_seqs = []    # per image: 1 = delta-net predicts attack, 0 = benign, nan = HoDS not ready
        self.first_detects = []  # per image: query index of first attack prediction or None

        self.img = None
        self.attack_q = 0
        self.seq = []
        print("GWAD results are saved in %s" % self.dir)

    def start_image(self, img):
        self.img = img
        self.attack_q = 0
        self.seq = []

    def log_query(self, query, t, screened, ds, pred, x):
        ds = '' if ds is None else '%.6f' % float(ds)
        self.timeline.writerow([self.img, query, t, int(screened), ds, '' if pred is None else pred])
        self.seq.append(np.nan if pred is None else float(pred != 'benign'))

        # dummy benign queries of adaptive attacks are not saved
        if t == 'attack':
            self.attack_q += 1
            if self.save_every > 0 and self.attack_q % self.save_every == 0:
                img = Image.fromarray(self.example_save.ready_img(x))
                img = img.resize((img.width*4, img.height*4), Image.NEAREST)
                os.makedirs(os.path.join(self.dir, 'queries'), exist_ok=True)
                img.save(os.path.join(self.dir, 'queries', 'img_%04d_q%05d.png' % (self.img, self.attack_q)))

    def end_sequence(self, first_detect):
        self.detect_seqs.append(self.seq)
        self.first_detects.append(first_detect)

    def end_image(self, x, adv, true_class, adv_class, l2, ratio, gwad_hx, first_detect, total_queries):
        self.end_sequence(first_detect)

        counts = [int(c) for c in gwad_hx]
        verdict = self.delta_classes[int(np.argmax(counts))] if sum(counts) > 0 else 'none'
        success = adv_class != true_class
        row = {
            'img': self.img,
            'true_class': self.data_classes[true_class],
            'adv_class': self.data_classes[adv_class],
            'attack_success': int(success),
            'l2': '%.4f' % l2,
            'dist_ratio': '%.4f' % ratio,
            'attack_queries': self.attack_q,
            'total_queries': total_queries,
            'first_detection': '' if first_detect is None else first_detect,
            'gwad_verdict': verdict,
        }
        row.update({'n_%s' % c: n for c, n in zip(self.delta_classes, counts)})
        self.summary_rows.append(row)

        # original | adversarial | perturbation
        x0 = self.example_save.ready_img(x)
        x1 = self.example_save.ready_img(adv)
        d = np.abs(x1.astype(np.float32) - x0.astype(np.float32))
        scale = 255 / max(d.max(), 1e-8)
        d = np.uint8(d * scale)

        fig = Figure(figsize=(9, 3.6))
        axes = fig.subplots(1, 3)
        titles = ['original: %s' % row['true_class'],
                  'adversarial: %s' % row['adv_class'],
                  '|perturbation| x%.0f' % scale]
        for ax, im, title in zip(axes, [x0, x1, d], titles):
            ax.imshow(im, interpolation='nearest')
            ax.set_title(title)
            ax.axis('off')
        detect = 'none' if first_detect is None else '@ query %d' % first_detect
        fig.suptitle('test img %d | attack %s | L2 %.3f\nGWAD: %s (%d/%d) | first detection %s'
                     % (self.img, 'success' if success else 'failed', l2,
                        verdict, max(counts), sum(counts), detect))
        fig.tight_layout()
        os.makedirs(os.path.join(self.dir, 'summary'), exist_ok=True)
        fig.savefig(os.path.join(self.dir, 'summary', 'img_%04d.png' % self.img), dpi=100)

    def finish(self):
        self.timeline_file.close()

        if self.summary_rows:
            with open(os.path.join(self.dir, 'summary.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(self.summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.summary_rows)

        if self.detect_seqs:
            self.plot_timeline()

    def plot_timeline(self):
        n = max(len(s) for s in self.detect_seqs)
        mat = np.full((len(self.detect_seqs), n), np.nan)
        for i, s in enumerate(self.detect_seqs):
            # delta-net only runs on unscreened queries, so hold the last prediction (GWAD state) in between
            s = np.array(s)
            idx = np.where(np.isnan(s), 0, np.arange(len(s)))
            np.maximum.accumulate(idx, out=idx)
            mat[i, :len(s)] = s[idx]
        valid = np.sum(~np.isnan(mat), axis=0)
        rate = np.where(valid > 0, np.nansum(mat, axis=0) / np.maximum(valid, 1), np.nan)
        queries = np.arange(1, n + 1)

        fig = Figure(figsize=(9, 6))
        ax0, ax1 = fig.subplots(2, 1, sharex=True)
        ax0.plot(queries, rate)
        ax0.set_ylim(-0.05, 1.05)
        ax0.set_ylabel('predicted as attack\n(fraction of images)')
        ax0.set_title('%s : %d image(s)' % (self.name, len(self.detect_seqs)))
        ax0.grid(alpha=0.3)

        detects = [q for q in self.first_detects if q is not None]
        if detects:
            ax1.hist(detects, bins=np.linspace(0, n, 51))
        ax1.set_xlim(0, n)
        ax1.set_xlabel('query')
        ax1.set_ylabel('first detection\n(count)')
        ax1.text(0.99, 0.95, 'detected %d / %d, no detection %d'
                 % (len(detects), len(self.first_detects), len(self.first_detects) - len(detects)),
                 transform=ax1.transAxes, ha='right', va='top')
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.dir, 'gwad_timeline.png'), dpi=100)
