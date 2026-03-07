"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os

import torch
import torch.distributed as dist
from torch.hub import download_url_to_file, get_dir

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

# def download_cached_file(url, check_hash=True, progress=False):
#     """
#     Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
#     If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
#     """

#     def get_cached_file_path():
#         # a hack to sync the file path across processes
#         parts = torch.hub.urlparse(url)
#         filename = os.path.basename(parts.path)
#         cached_file = os.path.join(get_dir(), filename)

#         return cached_file

#     if is_main_process():
#         download_url_to_file(url, check_hash, progress)

#     if is_dist_avail_and_initialized():
#         dist.barrier()

#     return get_cached_file_path()

def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally using torch.hub.
    [cite: 11, 40]
    """
    def get_cached_file_path():
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(torch.hub.get_dir(), 'checkpoints', filename)
        return cached_file

    cached_file = get_cached_file_path()
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(cached_file), exist_ok=True)

    if is_main_process():
        # Only download if the file isn't already there
        if not os.path.exists(cached_file):
            print(f"Downloading: {url} to {cached_file}")
            # FIX: Arguments are (url, destination_path, hash_prefix, progress)
            torch.hub.download_url_to_file(url, cached_file, progress=progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return cached_file