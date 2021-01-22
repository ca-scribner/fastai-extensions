# Summary

A collection of widgets and extensions for fast.ai objects.

These are not necessarily all aimed at the same fastai version and purposefully do not import dependencies, but there are many.  See the code for more details.

# Tools

## ImageListS3:
* A fast.ai ImageList that reads images directly from S3 rather than from local files
* 

### Future/Todo
* fix parallelization problem below
* add local caching.  This would make the first pass through data slower, but future passes as fast as a regular ImageList!

### Notes/limitations:
* ImageListS3 currently does not work well in a DataBunch that has num_workers>1.  I think this is because the DataBunch tries to pkl the open Bucket object (which is attached/logged into a bucket), and Bucket does not handle that.  Maybe there's a way to handle that more gracefully.  Could change ImageListS3/open_image_from_s3 to expect S3 credentials/bucket name/key rather than instantiated Bucket/key, then it logs in itself.  Is there a downside?  Performance penalty might be balanced by throwing more workers at the problem

## s3ls

Tool to list files in s3 bucket