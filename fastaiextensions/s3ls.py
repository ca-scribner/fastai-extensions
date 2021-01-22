# From https://stackoverflow.com/questions/35803027/retrieving-subfolders-names-in-s3-bucket-from-boto3/51372405#51372405

import boto3
from collections import namedtuple
from operator import attrgetter


S3Obj = namedtuple('S3Obj', ['key', 'mtime', 'size', 'ETag'])


def s3ls(bucket, path, start=None, end=None, recursive=True, list_dirs=True,
           list_objs=True, limit=None):
    """
    Iterator that lists a bucket's objects under path, (optionally) starting with
    start and ending before end.

    If recursive is False, then list only the "depth=0" items (dirs and objects).

    If recursive is True, then list recursively all objects (no dirs).

    Args:
        bucket:
            a boto3.resource('s3').Bucket().
        path:
            a directory in the bucket.
        start:
            optional: start key, inclusive (may be a relative path under path, or
            absolute in the bucket)
        end:
            optional: stop key, exclusive (may be a relative path under path, or
            absolute in the bucket)
        recursive:
            optional, default True. If True, lists only objects. If False, lists
            only depth 0 "directories" and objects.
        list_dirs:
            optional, default True. Has no effect in recursive listing. On
            non-recursive listing, if False, then directories are omitted.
        list_objs:
            optional, default True. If False, then directories are omitted.
        limit:
            optional. If specified, then lists at most this many items.

    Returns:
        an iterator of S3Obj.

    Examples:
        # set up
        >>> s3 = boto3.resource('s3')
        ... bucket = s3.Bucket(name)

        # iterate through all S3 objects under some dir
        >>> for p in s3ls(bucket, 'some/dir'):
        ...     print(p)

        # iterate through up to 20 S3 objects under some dir, starting with foo_0010
        >>> for p in s3ls(bucket, 'some/dir', limit=20, start='foo_0010'):
        ...     print(p)

        # non-recursive listing under some dir:
        >>> for p in s3ls(bucket, 'some/dir', recursive=False):
        ...     print(p)

        # non-recursive listing under some dir, listing only dirs:
        >>> for p in s3ls(bucket, 'some/dir', recursive=False, list_objs=False):
        ...     print(p)
"""
    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
        # note: need to use a string just smaller than start, because
        # the list_object API specifies that start is excluded (the first
        # result is *after* start).
        kwargs.update(Marker=__prev_str(start))
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter='/')
        if not path.endswith('/'):
            path += '/'
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={'MaxItems': limit})
    paginator = bucket.meta.client.get_paginator('list_objects')
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if 'CommonPrefixes' in resp and list_dirs:
            q = [S3Obj(f['Prefix'], None, None, None) for f in resp['CommonPrefixes']]
        if 'Contents' in resp and list_objs:
            q += [S3Obj(f['Key'], f['LastModified'], f['Size'], f['ETag']) for f in resp['Contents']]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter('key'))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            # TODO: HACK.  My consumer downstream wants this to be a single text path, not the NamedTuple. 
            # Not sure how to pull that attribute out without making it into a list/remove the fact that
            # it is a generator. For now, just yield only the filename
#             yield p
            yield p.key


def __prev_str(s):
    if len(s) == 0:
        return s
    s, c = s[:-1], ord(s[-1])
    if c > 0:
        s += chr(c - 1)
    s += ''.join(['\u7FFF' for _ in range(10)])
    return s
