# %%
from ast import Str
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

################################################## paths #########################################################
BASE_DIR='../working_dir'
MNI_brain=(opj(dirname(__file__),'../data/clean_data/templates/MNI152_T1_2mm_brain.nii.gz'))
MNI_brain=os.path.normpath(MNI_brain)
MNI=(opj(dirname(__file__),'../data/clean_data/templates/MNI152_T1_2mm.nii.gz'))
MNI=os.path.normpath(MNI)
brain_mask=(opj(dirname(__file__),'../data/clean_data/templates/brain_mask.nii.gz'))
brain_mask=os.path.normpath(brain_mask)
path_to_controls=(opj(dirname(__file__),'../data/IISASL_healthy_control_subjects/'))
path_to_controls=os.path.normpath(path_to_controls)
ASL_average=(opj(dirname(__file__),'../data/templates/ASL_average.nii.gz/'))
ASL_average=os.path.normpath(ASL_average)

################################################## argsparse #####################################################
parser = argparse.ArgumentParser(description='IISASL args')
parser.add_argument('-s','--subjects', nargs='+',required=False,
                    help="specify single subject id(s) in the form of 'sub-00...'")
parser.add_argument('-k','--kernel', type=int,required=False,default=1,
                    help="specify the number of kernel available. Default 1, recommend 2 for a single subject and more for multiple processing'")
parser.add_argument('-input_path',required=False,
                    help="specify the path to the subject folders. If not specified, location in ../patients folder is assumed'")
parser.add_argument('-output_path',required=False,
                    help="specify the path to the subject folders. If not specified, location in ../patients folder is assumed'")
d = {'output_path':False,'input_path':False,'kernel':False,'subjects':False}
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
    path_to_output="../IISASL_output/"


################################################# Import files ###################################################
templates={'raw_ictal':'{subject_id}/SPECT/{subject_id}_raw_ictal.nii*','raw_interictal': '{subject_id}/SPECT/{subject_id}_raw_interictal.nii*','struct':'{subject_id}/anat/T1.nii*' }
infosource = Node(IdentityInterface(fields=['subject_id']),name="infosource", iterables=[('subject_id', subject_list)])
selectfiles = Node(SelectFiles(templates,base_directory=input_path, raise_on_empty=False),name="selectfiles")

##################################################################################################################
########################################### Preprocessing nodes ##################################################
##################################################################################################################
reorient_ictal=Node(fsl.Reorient2Std(out_file='reo_ictal.nii.gz'),name="reorient_ictal")
reorient_interictal=Node(fsl.Reorient2Std(out_file='reo_interictal.nii.gz'),name="reorient_interictal")
recon_all=Node(ReconAll(directive='all'), name='recon_all')
MRI_convert_T1=Node(MRIConvert(out_type='niigz'), name='MRI_convert_T1')
MRI_convert_brainmask=Node(MRIConvert(out_type='niigz'), name='MRI_convert_brainmask')

reorient_t1=Node(fsl.Reorient2Std(out_file='t1.nii.gz'),name="reorient_t1")
reorient_brain=Node(fsl.Reorient2Std(out_file='reo_brain.nii.gz'),name="reorient_brain")

resample_t1=Node(fsl.FLIRT(out_file='resample_t1.nii.gz', apply_isoxfm=2, apply_xfm= False ),name="resample_t1")
skullstrip = Node(fsl.BET(robust=True,mask=True,frac=0.4,out_file='struct_brain.nii.gz'), name="skullstrip")


interictal_2_t1=Node(MRICoreg(), name='interictal_2_t1')
ictal_2_t1=Node(MRICoreg(), name='ictal_2_t1')
reorient_ictal_reg=Node(fsl.Reorient2Std(out_file='reo_ictal_reg.nii.gz'),name="reorient_ictal_reg")
reorient_interictal_reg=Node(fsl.Reorient2Std(out_file='reo_interictal_reg.nii.gz'),name="reorient_interictal_reg")
flirt_T1_2_MNI=Node(fsl.FLIRT(reference=MNI_brain), name='flirt_T1_2_MNI')
fnirt_T1_2_MNI=Node(fsl.FNIRT(fieldcoeff_file=True,ref_file=MNI), name='fnirt_T1_2_MNI')
concat_LTA_ictal=Node(ConcatenateLTA(), name='concat_LTA_ictal')
concat_LTA_interictal=Node(ConcatenateLTA(), name='concat_LTA_interictal')
convert_LTA_ictal=Node(LTAConvert(out_fsl=True), name='convert_LTA_ictal')
convert_LTA_interictal=Node(LTAConvert(out_fsl=True), name='convert_LTA_interictal')
concat_ictal = Node(fsl.ConvertXFM(concat_xfm=True),name='concat_ictal')
concat_interictal = Node(fsl.ConvertXFM(concat_xfm=True),name='concat_interictal')
apply_flirt_ictal=Node(fsl.ApplyXFM(apply_xfm=True), name='apply_flirt_ictal')
apply_flirt_interictal=Node(fsl.ApplyXFM(apply_xfm=True), name='apply_flirt_interictal')
apply_fnirt_ictal=Node(fsl.ApplyWarp(ref_file=MNI), name='apply_fnirt_ictal')
apply_fnirt_interictal=Node(fsl.ApplyWarp(ref_file=MNI), name='apply_fnirt_interictal')


filter_noise_ictal=Node(fsl.Threshold(thresh=5, use_robust_range=True, use_nonzero_voxels=False), name='filter_noise_ictal')
mask_low_thresh_ictal=Node(fsl.Threshold(thresh=5, use_robust_range=True, use_nonzero_voxels=False), name='mask_low_thresh_ictal')
bin_low_thresh_ictal=Node(fsl.UnaryMaths(operation='bin'), name='bin_low_thresh_ictal')
filter_noise_interictal=Node(fsl.Threshold(thresh=5, use_robust_range=True, use_nonzero_voxels=False), name='filter_noise_interictal')
mask_low_thresh_interictal=Node(fsl.Threshold(thresh=5, use_robust_range=True, use_nonzero_voxels=False), name='mask_low_thresh_interictal')
bin_low_thresh_interictal=Node(fsl.UnaryMaths(operation='bin'), name='bin_low_thresh_interictal')
combine_low_thresh_masks=Node(fsl.BinaryMaths(operation='mul'), name='combine_low_thresh_masks')
apply_low_thresh_masks=Node(fsl.BinaryMaths(operation='mul' ),name="apply_low_thresh_masksl")
apply_mask_warped_ictal=Node(fsl.BinaryMaths(operation='mul',operand_file=brain_mask ),name="warped_masked_ictal")
apply_mask_warped_interictal=Node(fsl.BinaryMaths(operation='mul',operand_file=brain_mask),name="warped_masked_interictal")
smooth_sub=Node(fsl.IsotropicSmooth(fwhm=16, out_file='smooth_sub.nii.gz'), name='smoothing_sub')
subtraction=Node(fsl.BinaryMaths(operation='sub',out_file='raw_ictal_interictal_difference.nii.gz'), name='subtraction') 

def histomatch(in_file):
    import os
    import nibabel as nib
    from skimage.exposure import match_histograms
    ASL_average=(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../../../../data/templates/ASL_average.nii.gz/'))
    ASL_average=os.path.normpath(ASL_average)
    ref = nib.load(ASL_average)
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
def histomatch2(in_file):
    import os
    import nibabel as nib
    from skimage.exposure import match_histograms
    ASL_average=(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../../../../data/templates/ASL_average.nii.gz/'))
    ASL_average=os.path.normpath(ASL_average)
    ref = nib.load(ASL_average)
    img = nib.load(in_file)
    img_array=img.get_fdata()
    ref_array=ref.get_fdata()
    affine=img.affine
    matched = match_histograms(img_array, ref_array)
    new_image = nib.Nifti1Image(matched, affine)
    filename = os.path.join(os.getcwd(), 'histomatched_interictal.nii.gz')
    new_image.to_filename(filename)
    return filename
histomatch2 = Node(Function(input_names=['in_file'], output_names=["out_file"], function=histomatch2), name="histomatch2")

def concatenate_2_files(file1, file2):
   return [file1, file2]

def sum_of_length(string1,string2):
    return string1 + string2

def design_matrix(file1, file2):
    sum_of_controls=len(file1)+len(file2)
    print(sum_of_controls)
    ones=([1]*sum_of_controls)
    patient=[0]
    ones.extend(patient)
    print(ones)
    zeros=([0]*sum_of_controls)
    patient_1=[1]
    zeros.extend(patient_1)
    print(zeros)
    return dict(reg1=zeros, reg2=ones)

def return_first(file):
    return file[0]

IISASL_preprocess_wf=Workflow(name="IISASL_wf_preproc",  base_dir=(os.path.join(BASE_DIR, 'IISASL_wf')))
IISASL_preprocess_wf.connect([ 
                (infosource, selectfiles, [('subject_id', 'subject_id')]),
                (selectfiles,recon_all, [('struct', 'T1_files')]), 
                (infosource,recon_all, [('subject_id', 'subject_id')]),
                (selectfiles,reorient_ictal, [('raw_ictal', 'in_file')]), 
                (selectfiles,reorient_interictal, [('raw_interictal', 'in_file')]), 
                (reorient_ictal, filter_noise_ictal, [('out_file', 'in_file')]),
                (reorient_interictal, filter_noise_interictal, [('out_file', 'in_file')]),
                (selectfiles,skullstrip,[('struct','in_file')]), 

                (recon_all,MRI_convert_T1,[('T1','in_file')]), 
                (recon_all,MRI_convert_brainmask,[('brainmask','in_file')]), 
                (MRI_convert_T1,resample_t1,[('out_file','in_file')]), 
                (MRI_convert_T1,resample_t1,[('out_file','reference')]), 
                (resample_t1,reorient_t1,[('out_file','in_file')]), 
                (MRI_convert_brainmask,reorient_brain,[('out_file','in_file')]), 


                (filter_noise_interictal,interictal_2_t1, [('out_file', 'source_file')]),
                (filter_noise_ictal,ictal_2_t1, [('out_file', 'source_file')]),
                (infosource,ictal_2_t1, [('subject_id', 'subject_id')]),
                (infosource,interictal_2_t1, [('subject_id', 'subject_id')]),
                (ictal_2_t1, convert_LTA_ictal , [('out_lta_file', 'in_lta')]), 
                (convert_LTA_ictal, apply_flirt_ictal , [('out_fsl', 'in_matrix_file')]), 
                (MRI_convert_T1, apply_flirt_ictal , [('out_file', 'reference')]), 
                (filter_noise_ictal, apply_flirt_ictal , [('out_file', 'in_file')]), 

                (interictal_2_t1, convert_LTA_interictal , [('out_lta_file', 'in_lta')]), 
                (MRI_convert_T1, apply_flirt_interictal , [('out_file', 'reference')]), 
                (filter_noise_interictal, apply_flirt_interictal , [('out_file', 'in_file')]), 
                (convert_LTA_interictal, apply_flirt_interictal , [('out_fsl', 'in_matrix_file')]), 
                (reorient_brain, flirt_T1_2_MNI,[('out_file', 'in_file')]),
                (flirt_T1_2_MNI, fnirt_T1_2_MNI,[('out_matrix_file', 'affine_file')]),
                (reorient_t1, fnirt_T1_2_MNI,[('out_file', 'in_file')]),

                (apply_flirt_ictal , reorient_ictal_reg,[('out_file', 'in_file')]), 
                (apply_flirt_interictal , reorient_interictal_reg,[('out_file', 'in_file')]), 
                (reorient_ictal_reg, apply_fnirt_ictal,[('out_file', 'in_file')]),
                (fnirt_T1_2_MNI, apply_fnirt_ictal,[('fieldcoeff_file', 'field_file')]),
                (reorient_interictal_reg, apply_fnirt_interictal,[('out_file', 'in_file')]),
                (fnirt_T1_2_MNI, apply_fnirt_interictal,[('fieldcoeff_file', 'field_file')]),

                (apply_fnirt_ictal, apply_mask_warped_ictal, [('out_file', 'in_file' )]), 
                (apply_fnirt_interictal, apply_mask_warped_interictal, [('out_file', 'in_file' )]), 
                (apply_mask_warped_ictal, histomatch, [('out_file','in_file')]),
                (apply_mask_warped_interictal, histomatch2, [('out_file','in_file')]),
                (histomatch, mask_low_thresh_ictal, [('out_file', 'in_file')]),
                (histomatch2, mask_low_thresh_interictal, [('out_file', 'in_file')]),
                (mask_low_thresh_ictal, bin_low_thresh_ictal, [('out_file', 'in_file')]),
                (mask_low_thresh_interictal, bin_low_thresh_interictal, [('out_file', 'in_file')]),
                (bin_low_thresh_ictal, combine_low_thresh_masks, [('out_file', 'in_file')]),
                (bin_low_thresh_interictal, combine_low_thresh_masks, [('out_file', 'operand_file')]),
                (histomatch,subtraction, [('out_file', 'in_file')] ),
                (histomatch2,subtraction, [('out_file', 'operand_file')] ),
                (subtraction, smooth_sub, [('out_file', 'in_file')])  ])
IISASL_preprocess_wf.write_graph("IISASL_wf_preproc.dot",graph2use='colored')

###############################################################################################################
############################################## defining nodes #################################################
###############################################################################################################

templates={'warped_asl': 'sub*/normalized_differences.nii.gz', 'warped_spect':'sub*/normalized_differences.nii.gz'}
select_control_files = Node(SelectFiles(templates,base_directory=path_to_controls),name="select_healhty_control_files")
merge_controls_asl=Node(fsl.Merge(dimension='t'), name='merge_asl')
merge_controls_spect=Node(fsl.Merge(dimension='t'), name='merge_spect')
merge_controls = Node(fsl.Merge(dimension='t'), name='merge_asl_spect_controls')
merge_controls_patient = Node(fsl.Merge(dimension='t'), name='merge_controls_patient')
merge_subject=Node(fsl.Merge(dimension='t', merged_file='merged_files.nii.gz'), name='merge_subject')
absolute_value_subjects=Node(fsl.UnaryMaths(operation='abs', out_file='abs_merged_files.nii.gz'), name='absolute_value_subjects')
threshold_hyperperfusion=Node(fsl.Threshold(thresh=0), name='threshold_hyperperfusion')
threshold_hypoperfusion=Node(fsl.Threshold(thresh=0, direction='above'), name='threshold_hypoperfusion')
mask_threshold_hyperperfusion=Node(fsl.UnaryMaths(operation='bin'), name='mask_threshold_hyperperfusion')
mask_threshold_hypoperfusion=Node(fsl.UnaryMaths(operation='bin'), name='mask_threshold_hypoperfusion')
inv_threshold_hypoperfusion=Node(fsl.BinaryMaths(operation='mul', operand_value=-1), name='inv_threshold_hypoperfusion')

hyperperfusion_randomize_t_corrected=Node(fsl.ApplyMask(out_file='hyperperfusion_tfce.nii.gz'), name='hyperperfusion_randomize_t_corrected')
hyperperfusion_randomize_tstats=Node(fsl.ApplyMask(out_file='hyperperfusion_raw_tstats.nii.gz'), name='hyperperfusion_randomize_tstats')
hypoperfusion_randomize_t_corrected=Node(fsl.ApplyMask(out_file='hypoperfusion_tfce.nii.gz'), name='hypoperfusion_randomize_t_corrected')
hypoperfusion_randomize_tstats=Node(fsl.ApplyMask(out_file='hypoperfusion_raw_tstats.nii.gz'), name='hypoperfusion_randomize_tstats')

select_corrected_tstat=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_corr")
select_corrected_tstat_2=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_corr_2")
select_tstat=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat")
select_tstat_2=Node(Function(input_names=["file"], output_names=["output"], function=return_first), name="return_tstat_2")
sum_of_lengths = Node(Function(input_names=["string1", "string2"], output_names=["length"], function=sum_of_length), name="sum_of_lengths")
glm_dict = Node(Function(input_names=["file1", "file2"], output_names=["dict"], function=design_matrix), name="design_matrix")
select_first_tstat=Node(Function(input_names=["in_files"], output_names=["one_file"]), name="select_tstat" )
select_first_corrected=Node(Function(input_names=["in_files"], output_names=["one_file"]), name="select_corrected" )
concatenate = Node(Function(input_names=["file1", "file2"], output_names=["out_files"], function=concatenate_2_files), name="concat_files")
concatenate_merged_files = Node(Function(input_names=["file1", "file2"], output_names=["out_files"], function=concatenate_2_files), name="concat_merged_files")
combine_masks=Node(fsl.BinaryMaths(operation='mul'), name='combine_masks')
combine_norm_masks=Node(fsl.BinaryMaths(operation='mul', operand_file=str(brain_mask)), name='combine_masks')
model = Node(fsl.MultipleRegressDesign(contrasts = [['sub1>sub2', 'T',['reg1', 'reg2'],[1,-1]],['sub2>sub1', 'T',['reg1','reg2'],[-1,1]]], ), name='GLM')
randomise=Node(fsl.Randomise( tfce=True, raw_stats_imgs=True ), name='randomise')
inverse_warp=Node(fsl.InvWarp(), name='inverse_warp')
merge_randomise = Node(Merge(4), name='merge_randomise_output')
apply_inverse_warp=MapNode(fsl.ApplyWarp(),iterfield=['in_file'], name='apply_inverse_warp')
apply_mask_mni_space=MapNode(fsl.BinaryMaths(operation='mul', operand_file=str(brain_mask)),iterfield=['in_file'], name='apply_mask_mni_space')
mask_native_output=MapNode(fsl.BinaryMaths(operation='mul' ),iterfield=['in_file'], name='mask_native_output')


Sink = Node(DataSink(), name='IISASL_output')
Sink.inputs.base_directory = os.path.join(path_to_output, 'IISASL_output')
Sink.inputs.substitutions = [('_subject_id_', 'sub-'),('_apply_mask_mni_space1',''),('_apply_inverse_warp1',''),('_apply_mask_mni_space2',''),('_apply_inverse_warp2',''),
('_apply_mask_mni_space3',''),('_apply_inverse_warp3',''),('_apply_mask_mni_space0',''),('_apply_inverse_warp0',''),('_mask_native_output0','') ,('_mask_native_output1','') 
,('_mask_native_output2','') ,('_mask_native_output3',''),('sub-sub-','sub-'), ('hyperperfusion_t_corrected_warp_maths.nii.gz', 'hyperperfusion_tfce_namename.nii.gz') ]

IISASL_wf=Workflow(name="IISASL_wf",  base_dir=BASE_DIR)
IISASL_wf.connect([  
                                (select_control_files, merge_controls_asl, [('warped_asl', 'in_files')]),
                                (select_control_files, merge_controls_spect, [('warped_spect', 'in_files')]),
                                (merge_controls_asl, concatenate_merged_files, [('merged_file', 'file1')]),
                                (merge_controls_spect, concatenate_merged_files, [('merged_file', 'file2')]),
                                (concatenate_merged_files, merge_controls,[('out_files', 'in_files')]),

                                (select_control_files, glm_dict,[('warped_asl', 'file1')]),
                                (select_control_files,glm_dict, [('warped_spect', 'file2')]),
                                (glm_dict,model,[('dict', 'regressors')]),
                                (model, randomise , [('design_mat', 'design_mat')]),
                                (model, randomise , [('design_con', 'tcon')]),
                                (merge_controls,concatenate,[('merged_file','file1')]),

                                (IISASL_preprocess_wf,concatenate,[('smoothing_sub.out_file','file2')]),
                                (concatenate,merge_controls_patient, [('out_files', 'in_files')] ),
                                (merge_controls_patient,absolute_value_subjects, [('merged_file', 'in_file')] ),
                                (IISASL_preprocess_wf,threshold_hyperperfusion,[('smoothing_sub.out_file','in_file')]),
                                (IISASL_preprocess_wf,threshold_hypoperfusion,[('smoothing_sub.out_file','in_file')]),
                                (threshold_hyperperfusion,mask_threshold_hyperperfusion, [('out_file','in_file')]),
                                (threshold_hypoperfusion, inv_threshold_hypoperfusion, [('out_file','in_file')]),
                                (inv_threshold_hypoperfusion, mask_threshold_hypoperfusion, [('out_file','in_file')]),
                                (absolute_value_subjects, randomise , [('out_file', 'in_file')]),
                                (IISASL_preprocess_wf, combine_norm_masks, [('combine_low_thresh_masks.out_file', 'in_file')]),
                                (combine_norm_masks, randomise, [('out_file', 'mask')]),

                                (mask_threshold_hyperperfusion, hyperperfusion_randomize_t_corrected, [('out_file', 'mask_file')]),
                                (mask_threshold_hypoperfusion, hypoperfusion_randomize_t_corrected, [('out_file', 'mask_file')]),
                                (mask_threshold_hyperperfusion, hyperperfusion_randomize_tstats, [('out_file', 'mask_file')]),
                                (mask_threshold_hypoperfusion, hypoperfusion_randomize_tstats, [('out_file', 'mask_file')]),

                                (randomise, select_corrected_tstat, [('t_corrected_p_files', 'file')]),
                                (randomise, select_corrected_tstat_2, [('t_corrected_p_files', 'file')]),
                                (randomise, select_tstat, [('tstat_files', 'file')]),
                                (randomise, select_tstat_2, [('tstat_files', 'file')]),

                                (select_corrected_tstat, hyperperfusion_randomize_t_corrected, [('output', 'in_file')]),
                                (select_corrected_tstat_2, hypoperfusion_randomize_t_corrected, [('output', 'in_file')]),
                                (select_tstat, hyperperfusion_randomize_tstats, [('output', 'in_file')]),
                                (select_tstat_2, hypoperfusion_randomize_tstats, [('output', 'in_file')]),
                                (hyperperfusion_randomize_t_corrected, merge_randomise, [('out_file', 'in1')]),
                                (hypoperfusion_randomize_t_corrected, merge_randomise, [('out_file', 'in2')]),
                                (hyperperfusion_randomize_tstats, merge_randomise, [('out_file', 'in3')]),
                                (hypoperfusion_randomize_tstats, merge_randomise, [('out_file', 'in4')]),
                                (IISASL_preprocess_wf, inverse_warp, [('fnirt_T1_2_MNI.fieldcoeff_file', 'warp')]),
                                (IISASL_preprocess_wf, inverse_warp, [('MRI_convert_T1.out_file', 'reference')]),
                                (inverse_warp,apply_inverse_warp , [('inverse_warp', 'field_file')]),
                                (IISASL_preprocess_wf,apply_inverse_warp, [('skullstrip.out_file','ref_file' )]),
                                (merge_randomise,apply_inverse_warp , [('out', 'in_file')]),
                                (IISASL_preprocess_wf, mask_native_output, [('skullstrip.mask_file','operand_file' )]),
                                (apply_inverse_warp, mask_native_output, [('out_file','in_file' )]),
                                (merge_randomise, apply_mask_mni_space, [('out', 'in_file')]),

                                (IISASL_preprocess_wf, Sink, [('reorient_t1.out_file', 'out1')]),
                                #(randomise, Sink, [('t_corrected_p_files', '@out2')]),
                                #(randomise, Sink, [('tstat_files', '@out3')]),
                                (IISASL_preprocess_wf, Sink, [('apply_fnirt_interictal.out_file', '@out4')]),
                                (IISASL_preprocess_wf, Sink, [('apply_fnirt_ictal.out_file', '@out5')]),
                                (IISASL_preprocess_wf, Sink, [('fnirt_T1_2_MNI.warped_file', '@out6')]),
                                (IISASL_preprocess_wf, Sink, [('smoothing_sub.out_file', '@out7')]),
                                (mask_native_output, Sink, [('out_file', '@out8')]),
])
IISASL_wf.write_graph("IISASL_wf.dot",graph2use='colored')
if args.kernel:
        print('Running multiproc...')
        IISASL_wf.run('MultiProc', plugin_args={'n_procs': kernel})
else:
        print('Running single proc...')
        IISASL_wf.run()#'MultiProc', plugin_args={'n_procs': multiprocessing.cpu_count()})


