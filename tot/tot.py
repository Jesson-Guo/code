import numpy as np
import json
import copy


def build_tot(config, class_to_idx):
    builder = ToTBuilder()
    labels = builder.build_tree(config.DATA.DATASET, class_to_idx)
    root = builder.load(labels, config.TOT_PATH)
    tot = ToT(config.MODEL.NUM_CLASSES, root)
    tot.reset()

    tot_config = {
        'n': tot.n_layer,
        'num_plans': 2,
        'num_coarses': [],
    }

    for i in range(tot.n_layer+1):
        tot_config['num_coarses'].append(len(tot.coarse_cache['thoughts'][i]))

    config.defrost()
    config.MODEL.TOT.NUM_GRAN = tot_config['n']
    config.MODEL.TOT.NUM_COARSES = tot_config['num_coarses']
    config.MODEL.TOT.NUM_VOCAB = sum(tot_config['num_coarses'])
    config.freeze()

    return tot


class Thought(object):
    def __init__(self, labels, feedback=2, parent=None, sibling=[], name="", layer=0) -> None:
        self.labels = labels
        self.feedback = feedback
        self.parent = parent
        self.sibling = sibling
        self.plans = {}
        self.name = name
        self.layer = layer
        self.child_tids = [[], []]

        self.tid = -1

    def is_valid(self):
        return self.feedback > 0

    def stop(self):
        return len(self.labels) == 1

    def add_child(self, num, t):
        if num not in self.plans:
            self.plans[num] = []
        self.plans[num].append(t)

    def set(self):
        for i, ts in self.plans.items():
            for t in ts:
                self.child_tids[i].append(t.tid)

    def update_parent(self, t):
        self.parent = t

    def to_dict(self):
        label_list = [k for k in self.labels.keys()]
        return {"feedback": self.feedback, "labels": str(label_list), "name": self.name, "layer": self.layer, "plans": {}}


class ToT:
    def __init__(self, num_classes=0, root=None) -> None:
        self.num_coarses = num_classes
        self.num_classes = num_classes
        self.root = root

        self.n_layer = 0
        self.thought_dict = {}
        self.coarse_cache = {}
        self.leaves_cache = {}
        self.leaves_cnt = np.zeros((num_classes))

    def __init_thought_dict(self):
        self.num_coarses = 0
        self.thought_dict[-1] = self.root

        ts = [[self.root]]
        n_coarse = {0: 1}
        self.tid2label = {}
        while len(ts):
            q = []
            tmp = ts.pop()
            while len(tmp):
                t = tmp.pop()
                if t.stop():
                    self.n_layer = max(self.n_layer, t.layer-1)
                    self.leaves_cnt[list(t.labels.keys())[0]] += 1
                    self.tid2label[t.tid] = list(t.labels.keys())[0]
                for _ in t.plans.values():
                    for child in _:
                        q.insert(0, child)
                        child.tid = self.num_coarses
                        self.thought_dict[self.num_coarses] = child
                        self.num_coarses += 1
            if q == []:
                continue
            ts.append(q)
            n_coarse[t.layer+1] = len(q)

        self.tid2label_np = np.array(sorted(self.tid2label.items()))[:, 1]

        for _, t in self.thought_dict.items():
            t.set()
        thought_sets = {0: self.root.child_tids}
        for l in range(1, self.n_layer):
            thought_sets[l] = []
            sets = thought_sets[l-1]
            for s in sets:
                temp = [[], []]
                for i in range(2):
                    for tid in s:
                        t = self.thought_dict[tid]
                        temp[i].extend(t.child_tids[i])
                thought_sets[l].extend(temp)
        self.thought_sets = {}
        for l in range(self.n_layer):
            for s in thought_sets[l]:
                for tid in s:
                    self.thought_sets[tid] = s

    def __init_coarse_cache(self):
        self.coarse_cache["thoughts"] = {i: [] for i in range(self.n_layer+1)}
        self.coarse_cache["label2tid"] = []
        self.coarse_cache["max_child"] = [0 for _ in range(self.n_layer)]
        self.coarse_cache["coarse_labels"] = [[] for _ in range(self.n_layer)]
        self.coarse_cache["tid_mask"] = [[] for _ in range(self.n_layer)]
        self.coarse_cache["tid2mask"] = [{} for _ in range(self.n_layer)]

        ts = [[self.root]]
        while len(ts):
            q = []
            tmp = ts.pop()
            while len(tmp):
                t = tmp.pop()
                for _ in t.plans.values():
                    for child in _:
                        q.insert(0, child)
            if q == [] or q[0].stop():
                continue
            ts.append(q)

            label2tid = [[] for _ in range(self.num_classes)]
            idx = 0
            for tt in q:
                for l in tt.labels.keys():
                    label2tid[l].insert(0, tt.tid)
                    idx += 1
            self.coarse_cache["label2tid"].append(label2tid)

        for i in range(self.n_layer):
            self.coarse_cache["label2tid"][i] = np.array(self.coarse_cache["label2tid"][i])

        for _, t in self.thought_dict.items():
            if not t.stop():
                self.coarse_cache["thoughts"][t.layer].append(t.tid)

        for l in range(self.n_layer):
            for tid in self.coarse_cache["thoughts"][l]:
                t = self.thought_dict[tid]
                self.coarse_cache["max_child"][l] = max(
                    self.coarse_cache["max_child"][l], max(len(t.plans[0]), len(t.plans[1])))

                coarse_labels = [{}, {}]
                for p, ts in t.plans.items():
                    temp = [i for i in range(self.num_classes)]
                    for j in range(len(ts)):
                        for k in ts[j].labels.keys():
                            coarse_labels[p][k] = j
                            temp.remove(k)
                    for k in temp:
                        coarse_labels[p][k] = len(ts)
                coarse_labels[0] = np.array(sorted(coarse_labels[0].items()))[:, 1]
                coarse_labels[1] = np.array(sorted(coarse_labels[1].items()))[:, 1]
                self.coarse_cache["coarse_labels"][l].extend(coarse_labels)

        for l in range(self.n_layer):
            for tid in self.coarse_cache["thoughts"][l]:
                t = self.thought_dict[tid]
                for _ in t.child_tids:
                    mask = np.zeros((len(self.coarse_cache["thoughts"][l+1])))
                    for c in _:
                        mask[c-self.coarse_cache["label2tid"][l].min()] = 1
                    self.coarse_cache["tid_mask"][l].append(mask)
                    for c in _:
                        self.coarse_cache["tid2mask"][l][c-self.coarse_cache["label2tid"][l].min()] = len(self.coarse_cache["tid_mask"][l]) - 1

        for i in range(self.n_layer):
            self.coarse_cache["coarse_labels"][i] = np.array(self.coarse_cache["coarse_labels"][i])
            self.coarse_cache["tid_mask"][i] = np.array(self.coarse_cache["tid_mask"][i], dtype=np.bool8)
            self.coarse_cache["tid2mask"][i] = np.array(sorted(self.coarse_cache["tid2mask"][i].items()))[:, 1]

    def __init_coarse_targets(self):
        self.coarse_cache["label2coarse"] = {i: [[] for _ in range(self.num_classes)] for i in range(self.n_layer)}
        for i in range(self.num_classes):
            for l, tids in self.coarse_cache["thoughts"].items():
                if l >= self.n_layer:
                    continue
                for tid in tids:
                    t = self.thought_dict[tid]
                    coarse = [[], []]
                    for j, _ in t.plans.items():
                        for p in range(len(_)):
                            if i in t.plans[j][p].labels:
                                coarse[j].append(1)
                            else:
                                coarse[j].append(0)
                    self.coarse_cache["label2coarse"][t.layer][i].append(coarse)

    def __init_leaves_cache(self):
        self.leaves_cache["label2pos"] = []
        self.leaves_cache["pos2label"] = []
        self.leaves_cache["label2tid"] = [[] for _ in range(self.num_classes)]

        for l in range(self.n_layer-1):
            q = []
            for tid in self.coarse_cache["thoughts"][l]:
                t = self.thought_dict[tid]
                for _ in t.plans.values():
                    for child in _:
                        q.insert(0, child)

            label2pos = [[] for _ in range(self.num_classes)]
            pos2label = {}
            idx = 0
            for tt in q:
                for l in tt.labels.keys():
                    label2pos[l].append(idx)
                    pos2label[idx] = l
                    idx += 1
            self.leaves_cache["label2pos"].append(label2pos)
            self.leaves_cache["pos2label"].append(pos2label)

        for i in range(self.n_layer-1):
            self.leaves_cache["label2pos"][i] = np.array(self.leaves_cache["label2pos"][i])
            self.leaves_cache["pos2label"][i] = np.array(sorted(self.leaves_cache["pos2label"][i].items()))[:, 1]

        for _, t in self.thought_dict.items():
            t.set()
            if t.stop():
                self.leaves_cache["label2tid"][self.tid2label[t.tid]].append(t.tid)

        self.leaves_cache["label2tid"] = np.array(self.leaves_cache["label2tid"])
        self.leaves_cache["label2tid"] -= self.leaves_cache["label2tid"].min()

    def reset(self):
        self.__init_thought_dict()
        self.__init_coarse_cache()
        self.__init_coarse_targets()
        self.__init_leaves_cache()


class ToTBuilder:
    def build_tree(self, dataset, class_to_idx):
        class_to_idx_t = {}
        if dataset.startswith('imagenet'):
            from nltk.corpus import wordnet as wn
            for k, v in class_to_idx.items():
                name = wn.synset_from_pos_and_offset(k[0], int(k[1:])).name()
                class_to_idx_t[name] = v
            class_to_idx = class_to_idx_t
            del class_to_idx_t

        labels = {}
        for k, v in class_to_idx.items():
            labels[v] = k

        return labels

    def save(self, root, save_path):
        out = root.to_dict()
        que = [(root, out)]

        while len(que):
            t, t_dict = que.pop()
            if len(t.plans) == 0:
                continue
            for i, plan in t.plans.items():
                t_dict["plans"][i] = []
                for t_c in plan:
                    t_c_dict = t_c.to_dict()
                    t_dict["plans"][i].append(t_c_dict)
                    que.insert(0, (t_c, t_c_dict))

        out = json.dumps(out, indent=4, separators=(',', ': '))
        f = open(save_path, 'w')
        f.write(out)
        f.close()

    def load(self, labels, load_path):
        def load_child(t_dict, layer):
            assert t_dict["labels"].startswith('[') or t_dict["labels"].endswith(']'), "please check your json file."
            assert not t_dict["labels"] == "[]", "labels is empty, please check your json file."

            label_list = t_dict["labels"][1:-1].split(',')
            label_list = [int(l.strip()) for l in label_list]
            label_list.sort()
            label_dict = {l: labels[l] for l in label_list}

            t = Thought(labels=label_dict, feedback=t_dict["feedback"], name=t_dict["name"], layer=layer)
            if "plans" not in t_dict:
                t_dict["plans"] = {}
            for i, ts in t_dict["plans"].items():
                for j in range(len(ts)):
                    child = load_child(ts[j], layer+1)
                    child.parent = t
                    t.add_child(int(i), child)
            return copy.deepcopy(t)

        f = open(load_path, 'r')
        tot_data = json.load(f)
        root = load_child(tot_data, 0)
        return root
