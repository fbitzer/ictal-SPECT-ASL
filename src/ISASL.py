# %%
from nipype.interfaces.utility import IdentityInterface, Function, Merge, Rename
from nipype.interfaces.freesurfer import MRICoreg, ConcatenateLTA, ReconAll, MRIConvert
from nipype.interfaces.freesurfer.utils import LTAConvert 
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine.workflows import Workflow
from nipype import Node, Function, MapNode
from os.path import join as opj, dirname
from nipype.interfaces import fsl
from types import SimpleNamespace

import argparse
import os

################################################## paths #####################################################
BASE_DIR='../working_dir'
MNI_brain=(opj(dirname(__file__),'../data/clean_data/templates/MNI152_T1_2mm_brain.nii.gz'))
MNI_brain=os.path.normpath(MNI_brain)
MNI=(opj(dirname(__file__),'../data/clean_data/templates/MNI152_T1_2mm.nii.gz'))
MNI=os.path.normpath(MNI)
brain_mask=(opj(dirname(__file__),'../data/clean_data/templates/brain_mask.nii.gz'))
brain_mask=os.path.normpath(brain_mask)
SPECT_mask=(opj(dirname(__file__),'../data/clean_data/templates/SPECT_mask.nii.gz'))
SPECT_mask=os.path.normpath(brain_mask)
SPECT_average=(opj(dirname(__file__),'../data/clean_data/templates/SPECT_average.nii.gz'))
SPECT_average=os.path.normpath(brain_mask)
path_to_controls=(opj(dirname(__file__),'../data/ISASL_healthy_control_subjects/'))
path_to_controls=os.path.normpath(path_to_controls)
################################################## argsparse #####################################################

parser = argparse.ArgumentParser(description='ISASL args')
parser.add_argument('-s','--subjects', nargs='+',required=False,
                    help="specify single subject id(s) in the form of 'sub-00...'")
parser.add_argument('-k','--kernel', type=int,required=False,default=1,
                    help="specify the number of kernel available. Default 1, recommend 2 for a single subject and more for multiple processing'")
parser.add_argument('-input_path',required=False,
                    help="specify the path to the subject folders. If not specified, location in ../patients folder is assumed'")
parser.add_argument('-output_path',required=False,
                    help="specify the path to the subject folders. If not specified, location in ../patients folder is assumed'")
d = {'output_path':False,'input_path':False,'kernel':False,'subject':False}
args = SimpleNamespace(**d)
args = parser.parse_args()

if args.kernel:
    kernel=args.kernel
else:
    kernel=1

if args.subjects:
    new_sublist = []
    for s in args.subjects:
        new_sublist.append(s)
    subject_list = new_sublist

if args.input_path:
    input_path=args.input_path

if args.output_path:
    path_to_output=args.output_path
else: 
    path_to_output="../ISASL_output/"


templates={'raw_ictal':'{subject_id}/SPECT/{subject_id}_raw_ictal.nii*','struct':'{subject_id}/anat/T1.nii*' }
infosource = Node(IdentityInterface(fields=['subject_id']),name="infosource", iterables=[('subject_id', subject_list)])
selectfiles = Node(SelectFiles(templates,base_directory=input_path, raise_on_empty=False),name="selectfiles")


##################################################################################################################
########################################### Preprocessing nodes ##################################################
##################################################################################################################

reorient_ictal=Node(fsl.Reorient2Std(out_file='reo_ictal.nii.gz'),name="reorient_ictal")
recon_all=Node(ReconAll(directive='all'), name='recon_all')
MRI_convert_T1=Node(MRIConvert(out_type='niigz'), name='MRI_convert_T1')
MRI_convert_brainmask=Node(MRIConvert(out_type='niigz'), name='MRI_convert_brainmask')

reorient_t1=Node(fsl.Reorient2Std(out_file='reo_t 1.nii.gz'),name="reorient_t1")
reorient_brain=Node(fsl.Reorient2Std(out_file='reo_brain.nii.gz'),name="reorient_brain")

resample_t1=Node(fsl.FLIRT(out_file='resample_t1.nii.gz', apply_isoxfm=2, apply_xfm= False ),name="resample_t1")
Robustfov=Node(fsl.RobustFOV(out_roi='struct_roi.nii.gz'),name="robust_fov")
skullstrip = Node(fsl.BET(robust=True,mask=True,frac=0.4,out_file='struct_brain.nii.gz'), name="skullstrip")
filter_noise_ictal=Node(fsl.Threshold(thresh=2, use_robust_range=True, use_nonzero_voxels=False), name='filter_noise_ictal')
mask_low_thresh_ictal=Node(fsl.Threshold(thresh=10, use_robust_range=True, use_nonzero_voxels=False), name='mask_low_thresh_ictal')
bin_low_thresh_ictal=Node(fsl.UnaryMaths(operation='bin'), name='bin_low_thresh_ictal')
apply_low_thresh_masks=Node(fsl.BinaryMaths(operation='mul' ),name="apply_low_thresh_masksl")
apply_flirt_ictal=Node(fsl.ApplyXFM(), name='apply_flirt_ictal')
convert_LTA_ictal=Node(LTAConvert(out_fsl=True), name='convert_LTA_ictal')
flirt_T1_2_MNI=Node(fsl.FLIRT(reference=MNI_brain), name='flirt_T1_2_MNI')
fnirt_T1_2_MNI=Node(fsl.FNIRT(fieldcoeff_file=True,ref_file=MNI), name='fnirt_T1_2_MNI')
fnirt_2_spect=Node(fsl.FNIRT(fieldcoeff_file=True,ref_file=SPECT_average), name='fnirt_2_spect')
ConvertWarps=Node(fsl.ConvertWarp(relwarp=True, reference=SPECT_average), name='ConvertWarps')
apply_fnirt_ictal=Node(fsl.ApplyWarp(ref_file=MNI), name='apply_fnirt_ictal')
apply_fnirt_ictal_2_spect=Node(fsl.ApplyWarp(ref_file=SPECT_average), name='apply_fnirt_ictal_2_spect')
ictal_2_t1=Node(MRICoreg(), name='ictal_2_t1')
reorient_ictal_reg=Node(fsl.Reorient2Std(out_file='reo_ictal_reg.nii.gz'),name="reorient_ictal_reg")
apply_mask_warped_ictal=Node(fsl.BinaryMaths(operation='mul',operand_file=brain_mask),name="warped_masked_ictal")
smooth_sub=Node(fsl.IsotropicSmooth(fwhm=16, out_file='smooth_sub.nii.gz'), name='smoothing_sub')
subtraction=Node(fsl.BinaryMaths(operand_file=SPECT_average ,operation='sub',out_file='difference.nii.gz'), name='subtraction') 

def histomatch(in_file):
    import os
    import nibabel as nib
    import numpy as np
    from skimage.exposure import match_histograms
    SPECT_average=(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../../../../data/templates/SPECT_average.nii.gz/'))
    SPECT_average=os.path.normpath(SPECT_average)
    ref = nib.load(SPECT_average)
    img = nib.load(in_file)
    img_array=img.get_fdata()
    ref_array=ref.get_fdata()
    affine=img.affine
    matched = match_histograms(img_array, ref_array)
    new_image = nib.Nifti1Image(matched, affine)
    filename = os.path.join(os.getcwd(), 'histomatched_ictal.nii.gz')
    new_image.to_filename(filename)
    return filename
histomatch = Node(Function(input_names=['in_file'], output_names=["out_file"], function=histomatch), name="histomatch")


def design_matrix(file1, file2):
    sum_of_controls=len(file1)+len(file2)
    ones=([1]*sum_of_controls)
    patient=[0]
    ones.extend(patient)
    zeros=([0]*sum_of_controls)
    patient_1=[1]
    zeros.extend(patient_1)
    return dict(reg1=zeros, reg2=ones)

def return_first(file):
    return file[0]

def concatenate_2_files(file1, file2):
   return [file1, file2]


###############################################################################################################
############################################# Preprocessing workflow ##########################################
###############################################################################################################

ISASL_preprocess_wf=Workflow(name="ISASL_preprocess_wf",  base_dir=(os.path.join(BASE_DIR, 'ISASL_wf')))
ISASL_preprocess_wf.connect([ 
                (infosource, selectfiles, [('subject_id', 'subject_id')]),
                (selectfiles,recon_all, [('struct', 'T1_files')]), 
                (infosource,recon_all, [('subject_id', 'subject_id')]),
                (selectfiles,reorient_ictal, [('raw_ictal', 'in_file')]), 
                (reorient_ictal, filter_noise_ictal, [('out_file', 'in_file')]),
                (selectfiles,skullstrip,[('struct','in_file')]), 
                (recon_all,MRI_convert_T1,[('T1','in_file')]), 
                (recon_all,MRI_convert_brainmask,[('brainmask','in_file')]), 
                (MRI_convert_T1,reorient_t1,[('out_file','in_file')]), 
                (MRI_convert_brainmask,reorient_brain,[('out_file','in_file')]), 
                (reorient_t1,resample_t1,[('out_file','reference')]), 
                (reorient_t1,resample_t1,[('out_file','in_file')]), 

                (filter_noise_ictal,ictal_2_t1, [('out_file', 'source_file')]),
                (infosource,ictal_2_t1, [('subject_id', 'subject_id')]),
                (ictal_2_t1, convert_LTA_ictal , [('out_lta_file', 'in_lta')]), 
                (convert_LTA_ictal, apply_flirt_ictal , [('out_fsl', 'in_matrix_file')]), 
                (reorient_t1, apply_flirt_ictal , [('out_file', 'reference')]), 
                (filter_noise_ictal, apply_flirt_ictal , [('out_file', 'in_file')]), 
                (apply_flirt_ictal , reorient_ictal_reg,[('out_file', 'in_file')]), 
                (reorient_brain, flirt_T1_2_MNI,[('out_file', 'in_file')]),
                (flirt_T1_2_MNI, fnirt_T1_2_MNI,[('out_matrix_file', 'affine_file')]),
                (reorient_t1, fnirt_T1_2_MNI,[('out_file', 'in_file')]),
                (fnirt_2_spect, ConvertWarps, [('fieldcoeff_file', 'warp2')]),
                (fnirt_T1_2_MNI, ConvertWarps, [('fieldcoeff_file', 'warp1')]),
                (convert_LTA_ictal, apply_fnirt_ictal,[('out_fsl', 'premat')]),
                (filter_noise_ictal, apply_fnirt_ictal,[('out_file', 'in_file')]),
                (fnirt_T1_2_MNI, apply_fnirt_ictal,[('fieldcoeff_file', 'field_file')]), 
                (apply_fnirt_ictal, fnirt_2_spect, [('out_file', 'in_file')]),

                (reorient_ictal_reg, apply_fnirt_ictal_2_spect,[('out_file', 'in_file')]), 
                (ConvertWarps, apply_fnirt_ictal_2_spect,[('out_file', 'field_file')]),
                (apply_fnirt_ictal_2_spect, apply_mask_warped_ictal, [('out_file', 'in_file')]),
                (apply_mask_warped_ictal, histomatch, [('out_file','in_file')]),
                (histomatch, mask_low_thresh_ictal, [('out_file', 'in_file')]),
                (mask_low_thresh_ictal, bin_low_thresh_ictal, [('out_file', 'in_file')]),
                (histomatch,subtraction, [('out_file', 'in_file')] ),
                (subtraction,apply_low_thresh_masks, [('out_file', 'in_file')] ),
                (bin_low_thresh_ictal,apply_low_thresh_masks, [('out_file', 'operand_file')] ),
                (apply_low_thresh_masks, smooth_sub, [('out_file', 'in_file')]) ])                                                                                                      
ISASL_preprocess_wf.write_graph("output_ISASL.dot",graph2use='colored')

###############################################################################################################
############################################# defining nodes ##################################################
###############################################################################################################

templates={'warped_asl': 'sub*/normalized_differences.nii.gz', 'warped_spect':'sub*/normalized_differences.nii.gz'}
select_control_files = Node(SelectFiles(templates,base_directory=path_to_controls),name="select_healhty_control_files")
merge_controls_asl=Node(fsl.Merge(dimension='t'), name='merge_asl')
merge_controls_spect=Node(fsl.Merge(dimension='t'), name='merge_spect')
merge_controls = Node(fsl.Merge(dimension='t'), name='merge_asl_spect_controls')
merge_controls_patient = Node(fsl.Merge(dimension='t'), name='merge_controls_patient')
select_corrected_tstat=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_corr")
select_corrected_tstat_2=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_corr_2")
select_tstat=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat")
select_tstat_2=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_2")
glm_dict = Node(Function(input_names=["file1", "file2"], output_names=["dict"], function=design_matrix), name="design_matrix")
select_first_tstat=Node(Function(input_names=["in_files"], output_names=["one_file"]), name="select_tstat" )
select_first_corrected=Node(Function(input_names=["in_files"], output_names=["one_file"]), name="select_corrected" )
concatenate = Node(Function(input_names=["file1", "file2"], output_names=["out_files"], function=concatenate_2_files), name="concat_files")
concatenate_merged_files = Node(Function(input_names=["file1", "file2"], output_names=["out_files"], function=concatenate_2_files), name="concat_merged_files")
combine_norm_masks=Node(fsl.BinaryMaths(operation='mul', operand_file=SPECT_mask), name='combine_masks')
model = Node(fsl.MultipleRegressDesign(contrasts = [['sub1>sub2', 'T',['reg1', 'reg2'],[1,-1]],['sub2>sub1', 'T',['reg1','reg2'],[-1,1]]], ), name='GLM')
randomise=Node(fsl.Randomise( tfce=True, vox_p_values=True, raw_stats_imgs=True), name='randomise')
cluster_randomise=Node(fsl.Cluster(threshold=0.027, out_index_file='thresh_cluster_index', out_localmax_txt_file='thresh_lmax.txt', out_size_file='out_size_file'), name='cluster_randomise')
inverse_warp=Node(fsl.InvWarp(), name='inverse_warp')
warp_average2t1 = Node(fsl.ConvertWarp(), name='warp_average2t1')
merge_randomise = Node(Merge(4), name='merge_randomise_output')
merge_randomise_2 = Node(Merge(4), name='merge_randomise_output_2')
apply_inverse_warp=MapNode(fsl.ApplyWarp(),iterfield=['in_file'], name='apply_inverse_warp')
apply_mask_mni_space=MapNode(fsl.BinaryMaths(operation='mul', operand_file=str(brain_mask)),iterfield=['in_file'], name='apply_mask_mni_space')
apply_flirt_native=MapNode(fsl.ApplyXFM(),iterfield=['in_file'], name='apply_warp_native')
mask_native_output=MapNode(fsl.BinaryMaths(operation='mul'),iterfield=['in_file'], name='mask_native_output')


Sink = Node(DataSink(), name='output_ISASL')
Sink.inputs.base_directory = os.path.join(path_to_output, 'ISASL_output')
Sink.inputs.substitutions = [('_subject_id_', 'sub-'),('_apply_mask_mni_space1',''),('_apply_inverse_warp1',''),('_apply_mask_mni_space2',''),('_apply_inverse_warp2',''),
('_apply_mask_mni_space3',''),('_apply_inverse_warp3',''),('_apply_mask_mni_space0',''),('_apply_inverse_warp0','')]

ISASL_wf=Workflow(name="ISASL_wf",  base_dir=BASE_DIR)
ISASL_wf.connect([  
                                (select_control_files, merge_controls_asl, [('warped_asl', 'in_files')]),
                                (select_control_files, merge_controls_spect, [('warped_spect', 'in_files')]),
                                (merge_controls_asl, concatenate_merged_files, [('merged_file', 'file1')]),
                                (merge_controls_spect, concatenate_merged_files, [('merged_file', 'file2')]),
                                (concatenate_merged_files, merge_controls,[('out_files', 'in_files')]),
                                (select_control_files, glm_dict,[('warped_asl', 'file1')]),
                                (select_control_files, glm_dict,[('warped_spect', 'file2')]),
                                (glm_dict,model,[('dict', 'regressors')]),
                                (model, randomise , [('design_mat', 'design_mat')]),
                                (model, randomise , [('design_con', 'tcon')]),
                                (merge_controls,concatenate,[('merged_file','file1')]),
                                (ISASL_preprocess_wf,concatenate,[('smoothing_sub.out_file','file2')]),
                                (concatenate,merge_controls_patient, [('out_files', 'in_files')] ),
                                (merge_controls_patient, randomise , [('merged_file', 'in_file')]),
                                (ISASL_preprocess_wf, combine_norm_masks, [('bin_low_thresh_ictal.out_file', 'in_file')]),
                                (combine_norm_masks, randomise, [('out_file', 'mask')]),
                                (randomise, select_corrected_tstat, [('t_corrected_p_files', 'file')]),
                                (randomise, select_corrected_tstat_2, [('t_corrected_p_files', 'file')]),
                                (randomise, select_tstat, [('tstat_files', 'file')]),
                                (randomise, select_tstat_2, [('tstat_files', 'file')]),
                                (select_corrected_tstat, merge_randomise, [('output', 'in1')]),
                                (select_corrected_tstat_2, merge_randomise, [('output', 'in2')]),
                                (select_tstat, merge_randomise, [('output', 'in3')]),
                                (select_tstat_2, merge_randomise, [('output', 'in4')]),
                                (ISASL_preprocess_wf, inverse_warp, [('fnirt_T1_2_MNI.fieldcoeff_file', 'warp')]),
                                (ISASL_preprocess_wf, inverse_warp, [('reorient_t1.out_file', 'reference')]),
                                (inverse_warp,apply_inverse_warp , [('inverse_warp', 'field_file')]),
                                (ISASL_preprocess_wf,apply_inverse_warp, [('reorient_t1.out_file','ref_file' )]),
                                (merge_randomise,apply_inverse_warp , [('out', 'in_file')]),
                                (ISASL_preprocess_wf, mask_native_output, [('skullstrip.mask_file','operand_file' )]),
                                (apply_inverse_warp, mask_native_output, [('out_file','in_file' )]),
                                (merge_randomise, apply_mask_mni_space, [('out', 'in_file')]),

                                (ISASL_preprocess_wf, Sink, [('apply_flirt_ictal.out_file', '@out1')]),
                                (ISASL_preprocess_wf, Sink, [('fnirt_T1_2_MNI.warped_file', '@out2')]),
                                (ISASL_preprocess_wf, Sink, [('fnirt_2_spect.warped_file', '@out3')]),
                                (ISASL_preprocess_wf, Sink, [('apply_fnirt_ictal.out_file', '@out4')])  
                                (apply_mask_mni_space, Sink, [('out_file', '@out5')]),
                                (mask_native_output, Sink, [('out_file', '@out6')]),
                                (ISASL_preprocess_wf, Sink, [('reorient_t1.out_file', 'out7')]),
                                (ISASL_preprocess_wf, Sink, [('smoothing_sub.out_file', 'out8')]),  ])  

ISASL_wf.write_graph("ISASL_wf.dot",graph2use='colored')

if args.kernel:
        print('Running multiproc...')
        ISASL_wf.run('MultiProc', plugin_args={'n_procs': kernel})
else:
        print('Running single proc...')
        ISASL_wf.run()#'MultiProc', plugin_args={'n_procs': multiprocessing.cpu_count()})





# %%
