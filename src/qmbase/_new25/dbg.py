# © 2025 Dmitry A. Konovalov — All rights reserved.
# File : dbg.py Created : 2025-06-27 at 10:27 am by Dmitry.A.Konovalov@gmail.com

# ------------------------------- DBG
# __DBG__ = True
__DBG__ = False

import numpy as np
import pandas as pd
import torch
import os, sys
import scipy.sparse
import scipy


def set_dbg(val):
    if val:
        pd.set_option('display.max_columns', None)  # or 1000
        pd.set_option('display.max_rows', None)  # or 1000
        pd.set_option('display.max_colwidth', None)  # or 199
    global __DBG__
    old_val = __DBG__
    __DBG__ = val
    return old_val


def get_dbg():
    global __DBG__
    return __DBG__


def dbg_stats(x):
    bad_idx = np.isnan(x)
    count_nans = np.sum(bad_idx)
    if count_nans > 0:
        dbg('count_nans')
        x = x[~bad_idx]
    dbg('np.mean(x)', 'np.std(x)')
    dbg('np.min(x)', 'np.max(x)')
    dbg('x.shape', 'x.dtype')

def dbg_sparse(x):
    try:
        shape = x.shape
        dtype = x.dtype
        nnz = x.nnz
        total = np.prod(shape)
        density = 100.0 * nnz / total if total else 0
        print(f"{shape} {dtype} nnz={nnz:,} ({density:.4f}%)", end='')

        if hasattr(x, 'data') and x.data.size > 0:
            vals = x.data
            print("  data: min=", np.min(vals), "mean=", np.mean(vals), "max=", np.max(vals))
        else:
            print("  data: <empty or not available>")
    except Exception as e:
        print("Error in dbg_sparse:", e)

# def dbg_sparse(x):
#     try:
#         shape = x.shape
#         dtype = x.dtype
#         nnz = x.nnz
#         shape_len = len(shape)
#         total = shape[0] * shape[1]
#         density = 100.0 * nnz / total if total else 0
#         print(f"{shape} {dtype} nnz={nnz:,} ({density:.4f}%)")
#
#         if isinstance(x, scipy.sparse.csr_matrix) or isinstance(x, scipy.sparse.coo_matrix):
#             vals = x.data
#             if vals.size > 0:
#                 print("  data: min=", np.min(vals), "mean=", np.mean(vals), "max=", np.max(vals))
#             else:
#                 print("  data: <empty>")
#     except Exception as e:
#         print("Error in dbg_sparse:", e)

def dbg_np(x):
    try:
        idx = np.isnan(x)
        count = np.sum(idx)
        if np.all(idx):
            print(x.shape, x.dtype, 'all_nans_count=', count)
            return
        if count == 0:
            print(x.shape, x.dtype, 'min=', np.min(x), 'mean=', np.mean(x), 'max=', np.max(x))
        else:
            # print(x.shape, x.dtype,
            #       'min=', np.nanmin(x), 'max=', np.nanmax(x), 'nan_count=', count)
            print(x.shape, x.dtype,
                  'min=', np.nanmin(x), 'mean=', np.nanmean(x), 'max=', np.nanmax(x), 'nan_count=', count)
    except:
        print(x.shape, x.dtype)


def dbgT(df, n=1):
    # dbg('df.head(n).T', 'df.tail(n).T', 'len(df)')
    dbg('list(df.columns)[:100]', 'list(df.dtypes)[:100]')
    dbg('df.tail(n).T', 'len(df)')


def display(a=None, **kwargs):
    print(a, **kwargs)


def dbg(x, x2=None, exp3=None, text=None, on=None):
    global __DBG__
    if on is None and not __DBG__:
        return __DBG__
    if on is not None and not on:
        return __DBG__
    if x is None:
        print('x=None')
    if isinstance(x, torch.HalfTensor) or isinstance(x, torch.Tensor):
        return dbg_pt(x)
    if isinstance(x, np.ndarray) and isinstance(x2, np.ndarray):
        dbg_np(x)
        dbg_np(x2)
        return
    if isinstance(x, np.ndarray):
        if x.dtype == complex:
            dbg_np(x.real)
            dbg_np(x.imag)
        else:
            dbg_np(x)
        return
    if scipy.sparse.issparse(x):
        dbg_sparse(x)
        return
    if isinstance(x, pd.DataFrame):
        print(list(x.columns)[:100])
        print(list(x.dtypes)[:100])
        # print(x.head(2))
        print(x.tail(1))
        print(x.tail(1).T)
        print(len(x))
        return

    if hasattr(x, 'items'):  # new 30-march-2021
        for k, v in x.items():
            print(f'{k}: {v}')
        return

    # if isinstance(x, list):
    #     print('list len=', len(x), x)
    #     return
    if isinstance(x, list):
        dbg('len(x)')
        for xi in x:
            dbg(xi)
        return

    if text:
        print(text)
    if isinstance(x, str):
        frame = sys._getframe(1)
        print(x, '=', repr(eval(x, frame.f_globals, frame.f_locals)))
    if x2 and isinstance(x2, str):
        print(x2, '=', repr(eval(x2, frame.f_globals, frame.f_locals)))
    if exp3 and isinstance(exp3, str):
        print(exp3, '=', repr(eval(exp3, frame.f_globals, frame.f_locals)))

    return __DBG__


def dbg_pt(x, x2_USE_LIST=None):
    if not get_dbg():
        return
    if x2_USE_LIST is not None:
        dbg_pt(x2_USE_LIST)
    if x is None:
        print('None')
    if isinstance(x, torch.HalfTensor):
        print(x.shape, x.dtype)
        return
    if isinstance(x, torch.Tensor):
        print(x.shape, x.dtype, 'min=', torch.min(x), 'max=', torch.max(x))
        return
    if isinstance(x, list):
        dbg('len(x)')
        for xi in x:
            dbg_pt(xi)
        return
    # dbg(x, x2)  # try dbg for others

