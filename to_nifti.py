import nrrd, numpy as np, nibabel as nib, gzip, shutil
with gzip.open("jaw_0p1403mm.nrrd.gz","rb") as f, open("jaw.nrrd","wb") as g: shutil.copyfileobj(f,g)
a,h=nrrd.read("jaw.nrrd")          # (x=cols R->L, y=rows A->P, z=slices I->S)
s=0.1403
aff=np.diag([-s,-s,s,1.0])         # NIfTI is RAS: +col = Left = -R, +row = Posterior = -A
nib.save(nib.Nifti1Image(a.astype(np.int16),aff),"jaw.nii.gz")
print(a.shape,a.dtype)
