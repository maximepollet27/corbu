import tqdm
import pickle
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

def build_mlp(input_dim, output_dim, num_layers, hidden_dim, activation,
              dropout, batch_norm):

    layers = []
    in_dim = input_dim

    for i in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(getattr(nn, activation)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, cond_dim, config):
        super(CVAE, self).__init__()

        # Extract architecture configs
        encoder_config = config["architecture"]["encoder"]
        decoder_config = config["architecture"]["decoder"]
        self.z_dim = encoder_config["z_dim"]

        # Build encoder q(z|x)
        self.encoder_net = build_mlp(
            input_dim=x_dim,
            output_dim=encoder_config["num_neurons"],
            num_layers=encoder_config["num_layers"],
            hidden_dim=encoder_config["num_neurons"],
            activation=encoder_config["activation_function"],
            dropout=encoder_config["dropout"],
            batch_norm=encoder_config["batch_norm"]
        )

        # Encoder heads
        self.y_head = nn.Linear(encoder_config["num_neurons"], y_dim)
        self.z_mu = nn.Linear(encoder_config["num_neurons"], self.z_dim)
        self.z_logvar = nn.Linear(encoder_config["num_neurons"], self.z_dim)

        # Build decoder p(x|y,z)
        self.decoder_net = build_mlp(
            input_dim=self.z_dim + cond_dim,
            output_dim=x_dim,
            num_layers=decoder_config["num_layers"],
            hidden_dim=decoder_config["num_neurons"],
            activation=decoder_config["activation_function"],
            dropout=decoder_config["dropout"],
            batch_norm=decoder_config["batch_norm"]
        )

        # Build conditional prior p(z|y)
        hidden_dim = 64
        self.prior_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_mu = nn.Linear(hidden_dim, self.z_dim)
        self.prior_logvar = nn.Linear(hidden_dim, self.z_dim)
    
    def encode(self, x):
        h = self.encoder_net(x)
        y_hat = self.y_head(h)
        mu_q = self.z_mu(h)
        logvar_q = self.z_logvar(h)
        return y_hat, mu_q, logvar_q

    def prior(self, y_condition):
        h = self.prior_net(y_condition)
        mu_p = self.prior_mu(h)
        logvar_p = self.prior_logvar(h)
        return mu_p, logvar_p

    def reparameterize(self, mu, logvar, tau=1.0):
        logvar = logvar.clamp(min=-10.0, max=6.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + tau * std * eps

    def decode(self, z, y_condition):
        z_y = torch.cat([z, y_condition], dim=1)
        return self.decoder_net(z_y)

    def forward(self, x, y_condition, tau=0.0):
        # 1. Encode x to predict performance y and
        # latent distribution N(mu_q, var_q)
        y_hat, mu_q, logvar_q = self.encode(x)

        # 2. Use conditional prior to predict latent distribution N(mu_p, var_p)
        mu_p, logvar_p = self.prior(y_condition)

        # 3. Sample z from latent space with reparametrization trick
        z = self.reparameterize(mu_q, logvar_q, tau)

        # 4. Decode to generate x from y and z
        x_hat = self.decode(z, y_condition)
        return x_hat, y_hat, mu_q, logvar_q, z, mu_p, logvar_p

def load_data(path, max_gwp, floor_area_target, device, min_gwp=150, num_gwp=10):
    # Load dataset artifacts (scalers, info, etc)
    with open(
        path.joinpath('./data/ml_model/data_objs.pkl'), "rb"
        ) as f:
        data_objs = pickle.load(f)
    X_scaler = data_objs["X_scaler"]
    y_scaler = data_objs["y_scaler"]
    cat_onehot_enc = data_objs["cat_onehot_enc"]
    disc_onehot_enc = data_objs["disc_onehot_enc"]
    cat_cols = data_objs.get("cat_cols", [])
    disc_cols = data_objs.get("disc_cols", [])
    conti_cols = data_objs.get("conti_cols", [])
    target_cols = data_objs.get("target_cols", [])
    n_cat  = sum(
        len(c) for c in (
            cat_onehot_enc.categories_ if cat_onehot_enc is not None else []
        )
    )
    n_disc = sum(
        len(c) for c in (
            disc_onehot_enc.categories_ if disc_onehot_enc is not None else []
        )
    )
    n_cont = len(conti_cols)

    # Gneerate gwp values to explore, and repeat floor area target
    gwp_values = np.linspace(
        min_gwp, max_gwp, num=num_gwp, dtype=np.float32,
    )
    floor_area_values = np.full_like(
        gwp_values, floor_area_target, dtype=np.float32
    )

    # Standardize conditions to training scale
    y_target_orig = np.stack(
        [gwp_values, floor_area_values],
        axis=1,
    ).astype(np.float32)
    y_target_scaled = torch.tensor(
        y_scaler.transform(y_target_orig),
        dtype=torch.float32,
        device=device,
    )

    return (
        X_scaler, y_scaler, cat_onehot_enc, disc_onehot_enc, cat_cols, disc_cols,
        conti_cols, target_cols, n_cat, n_disc, n_cont, y_target_orig,
        y_target_scaled
    )

def load_model(path, device):
    # Load trained model
    model_ckpt = torch.load(
        path.joinpath('./data/ml_model/model_ckpt.pt'),
        map_location=device
    )
    state_dict = model_ckpt.get("state_dict", model_ckpt)
    config = model_ckpt.get("config", None)
    model = CVAE(x_dim=65, y_dim=2, cond_dim=2, config=config).to(device)
    model.load_state_dict(state_dict)

    return model

def generate_designs(
        model, num_gen, tau, y_target_orig, y_target_scaled, X_scaler, y_scaler,
        cat_onehot_enc, disc_onehot_enc, n_cat, n_disc, n_cont, cat_cols,
        disc_cols, conti_cols, target_cols, device
    ):
    
    # generate building designs and back-transform it to the original space
    x_gen, y_rep = generate_from_y(model, y_target_scaled, num_gen, tau, device)
    x_enc = backtransform_X(
        x_gen, n_cat, n_disc, cat_onehot_enc, disc_onehot_enc, device
    )

    y_hat_std = surrogate_predict(model, x_enc)

    df = decode_X_to_df(
        x_gen, cat_onehot_enc, disc_onehot_enc, X_scaler, n_cat, n_disc, n_cont,
        cat_cols, disc_cols, conti_cols, y_hat_std, y_scaler, target_cols,
        y_target_orig, sort_by_error=True
    )

    return df

@torch.no_grad()
def generate_from_y(model, y_std, num_gen, tau, device):
    # Repeat y to have several generated samples
    y_std = y_std.to(device)
    y_rep = y_std.repeat_interleave(num_gen, dim=0)

    # use conditional prior to sample z
    mu_p, logvar_p = model.prior(y_rep)
    logvar_p = logvar_p.clamp(-10.0, 6.0)
    std_p = torch.exp(0.5 * logvar_p)
    z = mu_p + tau * std_p * torch.randn_like(std_p)

    # generate designs with decoder
    x_gen = model.decode(z, y_rep)
    return x_gen, y_rep

def backtransform_X(x_hat, n_cat, n_disc, cat_onehot_enc, disc_onehot_enc, device):

    B, D = x_hat.shape
    # Categorical columns
    if n_cat > 0 and cat_onehot_enc is not None:
        cat_logits = x_hat[:, :n_cat]
        cat_hard = torch.zeros_like(cat_logits, device=device)
        off = 0
        for cats in cat_onehot_enc.categories_:
            k = len(cats)
            sl = slice(off, off + k)
            # softmax over the group then argmax
            idx = F.softmax(cat_logits[:, sl], dim=1).argmax(dim=1)
            rows = torch.arange(B, device=device)
            cat_hard[rows, sl.start + idx] = 1.0
            off += k
    else:
        cat_hard = x_hat.new_zeros((B, 0))

    # Discrete columns
    if n_disc > 0 and disc_onehot_enc is not None:
        disc_logits = x_hat[:, n_cat:n_cat+n_disc]
        disc_hard = torch.zeros_like(disc_logits, device=device)
        off = 0
        for cats in disc_onehot_enc.categories_:
            k = len(cats)
            sl = slice(off, off + k)
            idx = F.softmax(disc_logits[:, sl], dim=1).argmax(dim=1)
            rows = torch.arange(B, device=device)
            disc_hard[rows, sl.start + idx] = 1.0
            off += k
    else:
        disc_hard = x_hat.new_zeros((B, 0))

    # Continuous columns
    cont_block = x_hat[:, n_cat+n_disc:] \
        if (n_cat + n_disc) < D else x_hat.new_zeros((B, 0))

    # concat 
    x_enc = torch.cat([cat_hard, disc_hard, cont_block], dim=1)
    return x_enc

@torch.no_grad()
def surrogate_predict(model, x: torch.Tensor):
    y_hat, _, _ = model.encode(x.to(next(model.parameters()).device))
    return y_hat

@torch.no_grad()
def decode_X_to_df(x_t, cat_onehot_enc, disc_onehot_enc, X_scaler, n_cat,
                   n_disc, n_cont, cat_cols, disc_cols, conti_cols, y_hat_std,
                   y_scaler, target_cols, y_target_orig, sort_by_error):

    X = x_t.detach().cpu().numpy()

    # 3 different blocks for each col type
    cat_block = X[:, :n_cat] if n_cat > 0 else np.zeros(
        (X.shape[0], 0), dtype=np.float32
    )
    disc_block = X[:, n_cat:n_cat+n_disc] if n_disc > 0 else np.zeros(
        (X.shape[0], 0), dtype=np.float32
    )
    cont_block = X[:, n_cat+n_disc:] if n_cont > 0 else np.zeros(
        (X.shape[0], 0), dtype=np.float32
    )

    # categorical one-hot
    if n_cat > 0 and cat_onehot_enc is not None:
        cat_hard = np.zeros_like(cat_block, dtype=np.float32)
        start = 0
        for cats in cat_onehot_enc.categories_:
            k = len(cats)
            sl = slice(start, start + k)
            idx = cat_block[:, sl].argmax(axis=1)
            rows = np.arange(cat_block.shape[0])
            cat_hard[rows, sl.start + idx] = 1.0
            start += k
        cats_decoded = cat_onehot_enc.inverse_transform(cat_hard)
        df_cat = pd.DataFrame(
            cats_decoded,
            columns=cat_cols if cat_cols else [
                f"cat_{i}" for i in range(len(cat_onehot_enc.categories_))
            ]
        )
    else:
        df_cat = pd.DataFrame(index=range(X.shape[0]))

    # discrete one-hot
    if n_disc > 0 and disc_onehot_enc is not None:
        disc_hard = np.zeros_like(disc_block, dtype=np.float32)
        start = 0
        for cats in disc_onehot_enc.categories_:
            k = len(cats)
            sl = slice(start, start + k)
            idx = disc_block[:, sl].argmax(axis=1)
            rows = np.arange(disc_block.shape[0])
            disc_hard[rows, sl.start + idx] = 1.0
            start += k
        disc_decoded = disc_onehot_enc.inverse_transform(disc_hard)
        df_disc = pd.DataFrame(
            disc_decoded,
            columns=disc_cols if disc_cols else [
                f"disc_{i}" for i in range(len(disc_onehot_enc.categories_))
            ]
        )
    else:
        df_disc = pd.DataFrame(index=range(X.shape[0]))

    # Inverse scale continuous 
    if n_cont > 0 and X_scaler is not None:
        cont_decoded = X_scaler.inverse_transform(cont_block)
        df_cont = pd.DataFrame(
            cont_decoded, columns=conti_cols if conti_cols else [
                f"cont_{i}" for i in range(n_cont)
            ]
        )
    else:
        df_cont = pd.DataFrame(index=range(X.shape[0]))

    # combine blocks
    df = pd.concat([df_cat.reset_index(drop=True),
                    df_disc.reset_index(drop=True),
                    df_cont.reset_index(drop=True)], axis=1)

    # add surrogate predictions (descaled)
    if y_hat_std is not None and y_scaler is not None:
        y_hat_orig = y_scaler.inverse_transform(
            y_hat_std.detach().cpu().numpy()
        )
        y_pred_cols = target_cols if \
            (target_cols and len(target_cols) == y_hat_orig.shape[1]) \
                else [f"y_pred_{i}" for i in range(y_hat_orig.shape[1])]
        for j, col in enumerate(y_pred_cols):
            df[col] = y_hat_orig[:, j]

    return df

def filter_designs(
        df, parcel_length, parcel_width, floor_area_target, max_gwp,
        floor_err=0.1, n_sol=15
    ):

    # Calc floor area and keep only designs within specified range
    df["floor_area"] = df["building_width"] * df["building_length"] \
        * df["nb_floors"]

    # Remove invalid designs
    df = df[
        (df["building_length"] <= parcel_length) \
            & (df["building_width"] <= parcel_width) \
            & (df["Mean GWP / Floor area"] <= max_gwp) \
            & (df["floor_area"] <= floor_area_target * (1+floor_err)) \
            & (df["floor_area"] >= floor_area_target * (1-floor_err))
    ]

    # Select diverse designs
    X_emb = make_embedding(
        df, ignore_cols=['Mean GWP / Floor area', 'floor_area']
    )
    idx_diverse = farthest_point_sampling(X_emb, k=n_sol, random_state=0)
    df_diverse = df.iloc[idx_diverse].reset_index(drop=True)
    return df_diverse

def make_embedding(df: pd.DataFrame, ignore_cols=None):
    if ignore_cols is None:
        ignore_cols = []

    # Columns to use for distance
    cols_for_distance = [c for c in df.columns if c not in ignore_cols]

    sub = df[cols_for_distance]

    # Split by type
    num_cols = sub.select_dtypes(include=[np.number]).columns.tolist()
    other_cols = [c for c in sub.columns if c not in num_cols]

    parts = []

    # Numeric
    if num_cols:
        X_num = sub[num_cols].astype(float).fillna(0.0).values
        parts.append(X_num)

    # Non-numeric encoded with factorize
    for col in other_cols:
        codes, _ = pd.factorize(sub[col], sort=False)
        parts.append(codes.astype(float)[:, None])

    X_full = np.hstack(parts).astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    return X_scaled

def farthest_point_sampling(X: np.ndarray, k: int, random_state: int = 0):
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    # Select first sample randomly
    first = rng.integers(n)
    selected = [first]

    # Calculate the euclidian distance between selected sample and all others
    dist_min = np.linalg.norm(X - X[first], axis=1)

    for _ in range(1, k):

        # Select the sample with the highest distance
        idx = int(np.argmax(dist_min))
        selected.append(idx)

        # Calculate distance between newly selected sample and all others
        dist_new = np.linalg.norm(X - X[idx], axis=1)

        # keep for each sample its closest distance to selected set
        dist_min = np.minimum(dist_min, dist_new)

    return np.array(selected, dtype=int)