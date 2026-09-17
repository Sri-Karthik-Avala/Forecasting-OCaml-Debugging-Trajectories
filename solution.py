# made by - Karthik
import sys, os, json, math, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier

START_TIME = time.time()
DEADLINE_SECONDS = 68 * 60

FAMILIES = ['name', 'syntax', 'typing', 'other', 'clear']
FAM_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}
N_FOLDS = 5
EMB_DIM = 40
PROG_HID = 40
DIAG_HID = 24
ATTEMPT_DIM = 64
TRAJ_HID = 64
GLOBAL_DIM = len(FAMILIES) + 3
POS_FEAT_DIM = 2
MAX_EPOCHS = 80
BATCH_SIZE = 32
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
MIN_FREQ = 2
PATIENCE = 10
REPAIR_LOSS_WEIGHT = 1.0
GBT_N_ESTIMATORS = 120
GBT_MAX_DEPTH = 2
GBT_LR = 0.05


def get_paths():
    if len(sys.argv) >= 3:
        public_dir = Path(sys.argv[1])
        submission_out = Path(sys.argv[2])
    else:
        public_dir = Path('dataset/public')
        submission_out = Path('working/submission.csv')
    return public_dir, submission_out


def elapsed():
    return time.time() - START_TIME


def time_left():
    return DEADLINE_SECONDS - elapsed()


# ---------------------------------------------------------------------------
# metric (matches the challenge's own eval formula, used only for local OOF)
# ---------------------------------------------------------------------------

def compute_score(learner_groups, y_risk, p_risk, y_repair, p_repair):
    learner_groups = np.asarray(learner_groups)
    y_risk = np.asarray(y_risk, dtype=np.float64)
    p_risk = np.clip(np.asarray(p_risk, dtype=np.float64), 0.0, 1.0)
    y_repair = np.asarray(y_repair, dtype=np.float64)
    p_repair = np.asarray(p_repair, dtype=np.float64)

    groups = np.unique(learner_groups)
    H = np.zeros(3)
    for h in range(3):
        group_H = []
        for g in groups:
            mask = learner_groups == g
            b = np.mean((p_risk[mask, h] - y_risk[mask, h]) ** 2)
            group_H.append(np.clip(1.0 - b / 0.25, 0.0, 1.0))
        H[h] = np.mean(group_H)
    S = float((H[0] * H[1] * H[2]) ** (1.0 / 3.0))

    u = 0.25
    E = np.sum((p_repair - y_repair) ** 2, axis=1)
    E0 = np.sum((u - y_repair) ** 2, axis=1)
    E0 = np.maximum(E0, 1e-12)
    R_i = np.clip(1.0 - E / E0, 0.0, 1.0)
    group_R = []
    for g in groups:
        mask = learner_groups == g
        group_R.append(np.mean(R_i[mask]))
    R = float(np.mean(group_R))

    A = (S + R) / 2.0
    Gs = math.sqrt(max(S * R, 0.0))
    score = 100.0 * (0.25 * A + 0.75 * Gs)
    return dict(H1=H[0], H2=H[1], H4=H[2], S=S, R=R, score=score)


# ---------------------------------------------------------------------------
# data loading / tokenization
# ---------------------------------------------------------------------------

def load_data(public_dir):
    train = pd.read_csv(public_dir / 'train.csv')
    test = pd.read_csv(public_dir / 'test.csv')
    train['trace'] = train['trace'].apply(json.loads)
    test['trace'] = test['trace'].apply(json.loads)
    train['repair_profile'] = train['repair_profile'].apply(json.loads)
    return train, test


def prog_tokens(text):
    return text.split(' ') if text else []


def diag_tokens(text):
    return text.split() if text else []


def build_vocab(train_df):
    counts = Counter()
    for trace in train_df['trace']:
        for att in trace:
            counts.update(prog_tokens(att['program']))
            counts.update(diag_tokens(att['diagnostic']))
    vocab = {'<pad>': 0, '<unk>': 1}
    for tok, c in counts.items():
        if c >= MIN_FREQ:
            vocab[tok] = len(vocab)
    return vocab


def encode_tokens(tokens, vocab):
    unk = vocab['<unk>']
    return [vocab.get(t, unk) for t in tokens]


def region_ids_for(n):
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    j = np.arange(n)
    r = np.minimum(3, (4 * j) // n)
    return r.astype(np.int64)


def build_structured_matrix(df):
    rows = []
    for _, row in df.iterrows():
        trace = row['trace']
        last = trace[-1]
        first = trace[0]
        ptoks = prog_tokens(last['program'])
        dtoks = diag_tokens(last['diagnostic'])
        feat = {
            'trace_len': len(trace),
            'last_repeat': last['repeat_count'],
            'n_tokens': len(ptoks),
            'diag_len': len(dtoks),
            'sum_repeat': sum(a['repeat_count'] for a in trace),
            'n_distinct_prog': len(set(a['program'] for a in trace)),
        }
        for f in FAMILIES:
            feat[f'fam_{f}'] = 1.0 if last['error_family'] == f else 0.0
        for f in FAMILIES:
            feat[f'firstfam_{f}'] = 1.0 if first['error_family'] == f else 0.0
        rows.append(feat)
    return pd.DataFrame(rows).values.astype(np.float64)


def build_examples(df, vocab, repeat_mean, repeat_std, ntok_mean, ntok_std,
                    distinct_mean, distinct_std, sumrep_mean, sumrep_std, is_train):
    examples = []
    for _, row in df.iterrows():
        trace = row['trace']
        first_fam_idx = FAM_TO_IDX.get(trace[0]['error_family'], FAM_TO_IDX['other'])
        n_distinct = len(set(a['program'] for a in trace))
        sum_repeat = sum(a['repeat_count'] for a in trace)
        log_distinct_n = (math.log1p(n_distinct) - distinct_mean) / distinct_std
        log_sumrep_n = (math.log1p(sum_repeat) - sumrep_mean) / sumrep_std
        global_struct = np.array([
            *[1.0 if k == first_fam_idx else 0.0 for k in range(len(FAMILIES))],
            log_distinct_n, log_sumrep_n, (len(trace) - 2.0) / 1.0,
        ], dtype=np.float32)
        attempts = []
        for pos, att in enumerate(trace):
            ptoks = prog_tokens(att['program'])
            dtoks = diag_tokens(att['diagnostic'])
            fam_idx = FAM_TO_IDX.get(att['error_family'], FAM_TO_IDX['other'])
            log_rep = math.log1p(att['repeat_count'])
            log_rep_n = (log_rep - repeat_mean) / repeat_std
            from_end = (len(trace) - 1 - pos) / 2.0
            is_last = 1.0 if pos == len(trace) - 1 else 0.0
            log_n = math.log1p(len(ptoks))
            log_n_n = (log_n - ntok_mean) / ntok_std
            struct = np.array([
                *[1.0 if k == fam_idx else 0.0 for k in range(len(FAMILIES))],
                log_rep_n, from_end, is_last, log_n_n,
            ], dtype=np.float32)
            n = len(ptoks)
            norm_pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)) if n > 0 else np.zeros(0, dtype=np.float32)
            reg = region_ids_for(n).astype(np.float32) / 3.0
            pos_feat = np.stack([norm_pos, reg], axis=1) if n > 0 else np.zeros((0, POS_FEAT_DIM), dtype=np.float32)
            attempts.append({
                'prog_ids': np.array(encode_tokens(ptoks, vocab), dtype=np.int64),
                'diag_ids': np.array(encode_tokens(dtoks, vocab), dtype=np.int64),
                'struct': struct,
                'pos_feat': pos_feat,
                'n_prog': len(ptoks),
            })
        last = attempts[-1]
        region_ids = region_ids_for(last['n_prog'])
        ex = {
            'case_id': row['case_id'],
            'learner_group': row['learner_group'],
            'attempts': attempts,
            'region_ids': region_ids,
            'n_last': last['n_prog'],
            'global_struct': global_struct,
        }
        if is_train:
            ex['y_risk'] = np.array([row['risk_1'], row['risk_2'], row['risk_4']], dtype=np.float32)
            ex['y_repair'] = np.array(row['repair_profile'], dtype=np.float32)
        examples.append(ex)
    return examples


STRUCT_DIM = len(FAMILIES) + 4


def collate(batch):
    B = len(batch)
    trace_lens = [len(ex['attempts']) for ex in batch]
    maxA = max(trace_lens)

    flat_prog, flat_diag, flat_struct, flat_pos_feat = [], [], [], []
    flat_row, flat_pos = [], []
    for bi, ex in enumerate(batch):
        for pos, att in enumerate(ex['attempts']):
            flat_prog.append(att['prog_ids'])
            flat_diag.append(att['diag_ids'])
            flat_struct.append(att['struct'])
            flat_pos_feat.append(att['pos_feat'])
            flat_row.append(bi)
            flat_pos.append(pos)
    N = len(flat_prog)

    maxT_prog = max(1, max(len(p) for p in flat_prog))
    maxT_diag = max(1, max(len(d) for d in flat_diag))

    prog_ids = np.zeros((N, maxT_prog), dtype=np.int64)
    prog_lens = np.ones(N, dtype=np.int64)
    prog_pos_feat = np.zeros((N, maxT_prog, POS_FEAT_DIM), dtype=np.float32)
    diag_ids = np.zeros((N, maxT_diag), dtype=np.int64)
    diag_lens = np.ones(N, dtype=np.int64)
    for i in range(N):
        p = flat_prog[i]
        if len(p) > 0:
            prog_ids[i, :len(p)] = p
            prog_lens[i] = len(p)
            prog_pos_feat[i, :len(p)] = flat_pos_feat[i]
        d = flat_diag[i]
        if len(d) > 0:
            diag_ids[i, :len(d)] = d
            diag_lens[i] = len(d)
    struct = np.stack(flat_struct, axis=0)

    gather_idx = np.zeros((B, maxA), dtype=np.int64)
    attempt_mask = np.zeros((B, maxA), dtype=np.float32)
    for i, (r, p) in enumerate(zip(flat_row, flat_pos)):
        gather_idx[r, p] = i
        attempt_mask[r, p] = 1.0

    last_flat_idx = np.zeros(B, dtype=np.int64)
    region_ids = np.full((B, maxT_prog), -1, dtype=np.int64)
    for bi, ex in enumerate(batch):
        last_flat_idx[bi] = gather_idx[bi, trace_lens[bi] - 1]
        n = ex['n_last']
        if n > 0:
            region_ids[bi, :n] = ex['region_ids']

    out = dict(
        prog_ids=torch.from_numpy(prog_ids),
        prog_lens=torch.from_numpy(prog_lens),
        prog_pos_feat=torch.from_numpy(prog_pos_feat),
        diag_ids=torch.from_numpy(diag_ids),
        diag_lens=torch.from_numpy(diag_lens),
        struct=torch.from_numpy(struct),
        gather_idx=torch.from_numpy(gather_idx),
        attempt_mask=torch.from_numpy(attempt_mask),
        trace_lens=torch.tensor(trace_lens, dtype=torch.int64),
        last_flat_idx=torch.from_numpy(last_flat_idx),
        region_ids=torch.from_numpy(region_ids),
        global_struct=torch.from_numpy(np.stack([ex['global_struct'] for ex in batch])),
    )
    if 'y_risk' in batch[0]:
        out['y_risk'] = torch.from_numpy(np.stack([ex['y_risk'] for ex in batch]))
        out['y_repair'] = torch.from_numpy(np.stack([ex['y_repair'] for ex in batch]))
    out['case_ids'] = [ex['case_id'] for ex in batch]
    out['learner_groups'] = [ex['learner_group'] for ex in batch]
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def masked_mean_max(out, lengths):
    B, T, H = out.shape
    device = out.device
    idx = torch.arange(T, device=device).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float().unsqueeze(-1)
    summed = (out * mask).sum(1)
    mean = summed / mask.sum(1).clamp(min=1.0)
    masked_out = out.masked_fill(mask == 0, float('-inf'))
    mx = masked_out.max(1).values
    mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    return mean, mx


class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid, need_token_scores=False, pos_feat_dim=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_feat_dim = pos_feat_dim
        self.rnn = nn.LSTM(emb_dim + pos_feat_dim, hid, batch_first=True, bidirectional=True)
        self.need_scores = need_token_scores
        if need_token_scores:
            self.token_score = nn.Linear(hid * 2 + pos_feat_dim, 1)

    def forward(self, ids, lengths, pos_feat=None):
        emb = self.emb(ids)
        inp = torch.cat([emb, pos_feat], dim=-1) if self.pos_feat_dim > 0 else emb
        packed = pack_padded_sequence(inp, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.shape[1])
        mean, mx = masked_mean_max(out, lengths)
        if self.need_scores:
            score_in = torch.cat([out, pos_feat], dim=-1) if self.pos_feat_dim > 0 else out
            scores = self.token_score(score_in).squeeze(-1)
        else:
            scores = None
        return mean, mx, scores


class TrajectoryModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.prog_enc = TokenEncoder(vocab_size, EMB_DIM, PROG_HID, need_token_scores=True, pos_feat_dim=POS_FEAT_DIM)
        self.diag_enc = TokenEncoder(vocab_size, EMB_DIM, DIAG_HID, need_token_scores=False)
        attempt_in = PROG_HID * 2 * 2 + DIAG_HID * 2 + STRUCT_DIM
        self.attempt_proj = nn.Sequential(
            nn.Linear(attempt_in, ATTEMPT_DIM), nn.ReLU(), nn.Dropout(0.2),
        )
        self.gru = nn.GRU(ATTEMPT_DIM, TRAJ_HID, batch_first=True)
        traj_dim = TRAJ_HID + ATTEMPT_DIM + GLOBAL_DIM
        self.risk_head = nn.Sequential(
            nn.Linear(traj_dim, 48), nn.ReLU(), nn.Dropout(0.2), nn.Linear(48, 3),
        )
        self.repair_bias = nn.Linear(traj_dim, 4)

    def forward(self, batch):
        prog_mean, prog_mx, prog_scores = self.prog_enc(batch['prog_ids'], batch['prog_lens'], batch['prog_pos_feat'])
        diag_mean, _, _ = self.diag_enc(batch['diag_ids'], batch['diag_lens'])
        attempt_feat = torch.cat([prog_mean, prog_mx, diag_mean, batch['struct']], dim=-1)
        attempt_repr = self.attempt_proj(attempt_feat)

        B, maxA = batch['gather_idx'].shape
        D = attempt_repr.shape[-1]
        seq = attempt_repr[batch['gather_idx'].reshape(-1)].reshape(B, maxA, D)
        seq = seq * batch['attempt_mask'].unsqueeze(-1)

        lengths = batch['trace_lens'].clamp(min=1)
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_last = h_n[-1]

        mask = batch['attempt_mask'].unsqueeze(-1)
        mean_pool = (seq * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        traj = torch.cat([h_last, mean_pool, batch['global_struct']], dim=-1)

        risk_logits = self.risk_head(traj)
        s = torch.sigmoid(risk_logits)
        risk_pred = torch.cumprod(s, dim=1)

        last_scores = prog_scores[batch['last_flat_idx']]
        region_ids = batch['region_ids']
        valid = (region_ids >= 0)
        region_ids_safe = region_ids.clamp(min=0)
        region_sum = torch.zeros(B, 4, device=seq.device)
        region_cnt = torch.zeros(B, 4, device=seq.device)
        scores_valid = torch.where(valid, last_scores, torch.zeros_like(last_scores))
        region_sum.scatter_add_(1, region_ids_safe, scores_valid)
        region_cnt.scatter_add_(1, region_ids_safe, valid.float())
        region_mean = region_sum / region_cnt.clamp(min=1.0)

        repair_logits = region_mean + self.repair_bias(traj)
        repair_pred = torch.softmax(repair_logits, dim=-1)

        return risk_pred, repair_pred


# ---------------------------------------------------------------------------
# train / infer
# ---------------------------------------------------------------------------

def run_epoch(model, examples, opt=None, batch_size=BATCH_SIZE):
    training = opt is not None
    model.train(training)
    order = np.random.permutation(len(examples)) if training else np.arange(len(examples))
    total_loss = 0.0
    n_batches = 0
    all_risk_pred, all_repair_pred = [], []
    all_case_ids = []
    for start in range(0, len(examples), batch_size):
        idx = order[start:start + batch_size]
        batch_examples = [examples[i] for i in idx]
        batch = collate(batch_examples)
        if training:
            opt.zero_grad()
            risk_pred, repair_pred = model(batch)
            risk_loss = F.mse_loss(risk_pred, batch['y_risk'])
            repair_loss = F.mse_loss(repair_pred, batch['y_repair'])
            loss = risk_loss + REPAIR_LOSS_WEIGHT * repair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        else:
            with torch.no_grad():
                risk_pred, repair_pred = model(batch)
            all_risk_pred.append(risk_pred.numpy())
            all_repair_pred.append(repair_pred.numpy())
            all_case_ids.extend(batch['case_ids'])
    if training:
        return total_loss / max(n_batches, 1)
    return np.concatenate(all_risk_pred), np.concatenate(all_repair_pred), all_case_ids


def predict(model, examples, batch_size=BATCH_SIZE):
    return run_epoch(model, examples, opt=None, batch_size=batch_size)


def build_submission_df(test_examples, test_risk, test_repair):
    test_risk = np.clip(test_risk, 0.0, 1.0)
    test_risk = np.minimum.accumulate(test_risk, axis=1)
    test_repair = np.clip(test_repair, 1e-6, None)
    test_repair = test_repair / test_repair.sum(axis=1, keepdims=True)

    rows = []
    for i, ex in enumerate(test_examples):
        rp = test_repair[i].tolist()
        s = sum(rp)
        if abs(s - 1.0) > 1e-9:
            rp = [v / s for v in rp]
        rows.append({
            'case_id': ex['case_id'],
            'risk_1': float(test_risk[i, 0]),
            'risk_2': float(test_risk[i, 1]),
            'risk_4': float(test_risk[i, 2]),
            'repair_profile': json.dumps(rp),
        })
    return pd.DataFrame(rows, columns=['case_id', 'risk_1', 'risk_2', 'risk_4', 'repair_profile'])


def write_submission(out_df, submission_out):
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = submission_out.with_suffix('.tmp.csv')
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, submission_out)


def main():
    public_dir, submission_out = get_paths()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    train_df, test_df = load_data(public_dir)
    vocab = build_vocab(train_df)

    all_repeats, all_ntoks = [], []
    for trace in train_df['trace']:
        for att in trace:
            all_repeats.append(math.log1p(att['repeat_count']))
            all_ntoks.append(math.log1p(len(prog_tokens(att['program']))))
    repeat_mean, repeat_std = float(np.mean(all_repeats)), float(np.std(all_repeats) + 1e-6)
    ntok_mean, ntok_std = float(np.mean(all_ntoks)), float(np.std(all_ntoks) + 1e-6)

    all_distinct, all_sumrep = [], []
    for trace in train_df['trace']:
        all_distinct.append(math.log1p(len(set(a['program'] for a in trace))))
        all_sumrep.append(math.log1p(sum(a['repeat_count'] for a in trace)))
    distinct_mean, distinct_std = float(np.mean(all_distinct)), float(np.std(all_distinct) + 1e-6)
    sumrep_mean, sumrep_std = float(np.mean(all_sumrep)), float(np.std(all_sumrep) + 1e-6)

    build_args = (vocab, repeat_mean, repeat_std, ntok_mean, ntok_std,
                  distinct_mean, distinct_std, sumrep_mean, sumrep_std)
    train_examples = build_examples(train_df, *build_args, is_train=True)
    test_examples = build_examples(test_df, *build_args, is_train=False)

    X_train_struct = build_structured_matrix(train_df)
    X_test_struct = build_structured_matrix(test_df)

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    if submission_out.exists():
        vdir = submission_out.parent
        existing = sorted(vdir.glob(submission_out.stem + '_v*.csv'))
        n = len(existing) + 1
        submission_out.rename(vdir / f'{submission_out.stem}_v{n}.csv')

    train_risk_mean = train_df[['risk_1', 'risk_2', 'risk_4']].mean().values
    train_repair_mean = np.stack(train_df['repair_profile'].values).mean(axis=0)
    train_repair_mean = train_repair_mean / train_repair_mean.sum()
    fallback_df = build_submission_df(
        test_examples,
        np.tile(train_risk_mean, (len(test_examples), 1)),
        np.tile(train_repair_mean, (len(test_examples), 1)),
    )
    write_submission(fallback_df, submission_out)
    print(f'wrote baseline fallback submission at t={elapsed():.0f}s', file=sys.stderr)

    groups = train_df['learner_group'].values
    n_groups = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    y_risk_all = np.stack([e['y_risk'] for e in train_examples])
    y_repair_all = np.stack([e['y_repair'] for e in train_examples])

    oof_risk = np.zeros((len(train_examples), 3))
    oof_repair = np.zeros((len(train_examples), 4))
    test_risk_sum = np.zeros((len(test_examples), 3))
    test_repair_sum = np.zeros((len(test_examples), 4))
    n_fold_models = 0

    oof_risk_gbt = np.zeros((len(train_examples), 3))
    test_risk_gbt_sum = np.zeros((len(test_examples), 3))
    n_gbt_models = 0

    fold_indices = list(gkf.split(np.zeros(len(train_examples)), groups=groups))

    for fold, (tr_idx, va_idx) in enumerate(fold_indices):
        remaining_folds = len(fold_indices) - fold
        per_fold_budget = time_left() / max(remaining_folds, 1)
        fold_deadline = elapsed() + per_fold_budget * 0.95
        if time_left() < 90:
            print(f'skip fold {fold}, out of time', file=sys.stderr)
            break

        tr_ex = [train_examples[i] for i in tr_idx]
        va_ex = [train_examples[i] for i in va_idx]

        model = TrajectoryModel(len(vocab))
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_score = -1.0
        best_state = None
        bad_epochs = 0
        for epoch in range(MAX_EPOCHS):
            if elapsed() > fold_deadline or time_left() < 60:
                break
            train_loss = run_epoch(model, tr_ex, opt=opt)
            risk_pred, repair_pred, _ = predict(model, va_ex)
            va_groups = np.array([e['learner_group'] for e in va_ex])
            y_risk = np.stack([e['y_risk'] for e in va_ex])
            y_repair = np.stack([e['y_repair'] for e in va_ex])
            m = compute_score(va_groups, y_risk, risk_pred, y_repair, repair_pred)
            if m['score'] > best_score:
                best_score = m['score']
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(f'fold {fold} epoch {epoch} loss {train_loss:.4f} val_score {m["score"]:.3f} '
                  f'S={m["S"]:.3f} R={m["R"]:.3f} best={best_score:.3f} t={elapsed():.0f}s', file=sys.stderr)
            if bad_epochs >= PATIENCE:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        risk_pred, repair_pred, va_case_ids = predict(model, va_ex)
        pos = {cid: i for i, cid in enumerate(va_case_ids)}
        order = [pos[e['case_id']] for e in va_ex]
        oof_risk[va_idx] = risk_pred[order]
        oof_repair[va_idx] = repair_pred[order]

        if len(test_examples) > 0:
            t_risk_pred, t_repair_pred, t_case_ids = predict(model, test_examples)
            pos_t = {cid: i for i, cid in enumerate(t_case_ids)}
            order_t = [pos_t[e['case_id']] for e in test_examples]
            test_risk_sum += t_risk_pred[order_t]
            test_repair_sum += t_repair_pred[order_t]
            n_fold_models += 1

        gbt_va_pred = np.zeros((len(va_idx), 3))
        gbt_test_pred = np.zeros((len(test_examples), 3))
        for h in range(3):
            y_tr_h = y_risk_all[tr_idx, h]
            if len(np.unique(y_tr_h)) < 2:
                gbt_va_pred[:, h] = float(y_tr_h.mean())
                gbt_test_pred[:, h] = float(y_tr_h.mean())
            else:
                clf = GradientBoostingClassifier(n_estimators=GBT_N_ESTIMATORS, max_depth=GBT_MAX_DEPTH,
                                                  learning_rate=GBT_LR, subsample=0.8, random_state=fold)
                clf.fit(X_train_struct[tr_idx], y_tr_h)
                gbt_va_pred[:, h] = clf.predict_proba(X_train_struct[va_idx])[:, 1]
                if len(test_examples) > 0:
                    gbt_test_pred[:, h] = clf.predict_proba(X_test_struct)[:, 1]
        gbt_va_pred = np.minimum.accumulate(gbt_va_pred, axis=1)
        gbt_test_pred = np.minimum.accumulate(gbt_test_pred, axis=1)
        oof_risk_gbt[va_idx] = gbt_va_pred
        if len(test_examples) > 0:
            test_risk_gbt_sum += gbt_test_pred
            n_gbt_models += 1

        print(f'fold {fold} done best_score={best_score:.3f} elapsed={elapsed():.0f}s', file=sys.stderr)

        if n_fold_models > 0:
            ckpt_df = build_submission_df(test_examples, test_risk_sum / n_fold_models, test_repair_sum / n_fold_models)
            write_submission(ckpt_df, submission_out)
            print(f'checkpoint: wrote submission after fold {fold} ({n_fold_models} models) at t={elapsed():.0f}s',
                  file=sys.stderr)

    oof_mask = np.any(oof_risk > 0, axis=1) | np.any(oof_repair > 0, axis=1)
    best_alpha = 1.0
    if oof_mask.sum() > 0:
        m = compute_score(groups[oof_mask], y_risk_all[oof_mask], oof_risk[oof_mask],
                           y_repair_all[oof_mask], oof_repair[oof_mask])
        print(f'FINAL OOF (NN only): {m}', file=sys.stderr)

        if n_gbt_models > 0:
            best_score_blend = -1.0
            for alpha in np.linspace(0.0, 1.0, 11):
                blended = alpha * oof_risk[oof_mask] + (1.0 - alpha) * oof_risk_gbt[oof_mask]
                blended = np.minimum.accumulate(blended, axis=1)
                mb = compute_score(groups[oof_mask], y_risk_all[oof_mask], blended,
                                    y_repair_all[oof_mask], oof_repair[oof_mask])
                print(f'blend alpha={alpha:.1f} score={mb["score"]:.3f} H1={mb["H1"]:.3f} '
                      f'H2={mb["H2"]:.3f} H4={mb["H4"]:.3f}', file=sys.stderr)
                if mb['score'] > best_score_blend:
                    best_score_blend = mb['score']
                    best_alpha = float(alpha)
            print(f'FINAL OOF best blend alpha={best_alpha:.2f} score={best_score_blend:.3f}', file=sys.stderr)

    if n_fold_models > 0:
        test_risk_nn = test_risk_sum / n_fold_models
        if n_gbt_models > 0:
            test_risk_gbt = test_risk_gbt_sum / n_gbt_models
            test_risk_final = best_alpha * test_risk_nn + (1.0 - best_alpha) * test_risk_gbt
        else:
            test_risk_final = test_risk_nn
        out_df = build_submission_df(test_examples, test_risk_final, test_repair_sum / n_fold_models)
        write_submission(out_df, submission_out)
        print(f'wrote final {len(out_df)} rows to {submission_out} at t={elapsed():.0f}s', file=sys.stderr)
    else:
        print(f'no fold completed in time; baseline fallback submission stands at t={elapsed():.0f}s', file=sys.stderr)


if __name__ == '__main__':
    main()
