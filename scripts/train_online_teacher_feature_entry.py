"""Teacher-layer comparison with evaluation-only semantic/exposure diagnostics."""
from pathlib import Path

base=Path('/content/train_online_calibration_fork_entry.py').read_text()
boundary="compiled=compile(source,'<online-calibration-fork>','exec')"
if base.count(boundary)!=1: raise RuntimeError('Unexpected fork entry boundary')
namespace=dict(__name__='feature_builder',__file__='/content/train_online_calibration_fork_entry.py')
exec(compile(base.split(boundary)[0],'<feature-builder>','exec'),namespace)
source=namespace['source']
anchor='            feature_otherep = torch.index_select(ft1, 0, r.view(-1))'
if source.count(anchor)!=1: raise RuntimeError('Unexpected target selection boundary')
source=source.replace(anchor,'''            weight = online_structure.known_weights(target_indices.cpu().numpy(),weight)
'''+anchor)
anchor='''    nomatch = online_structure.refresh(net, cls, optimizer_cls, epoch,
        reset_correspondence=(epoch == warmiter+1))'''
if source.count(anchor)!=1: raise RuntimeError('Unexpected teacher refresh boundary')
source=source.replace(anchor,anchor+'''
    if os.environ.get('ONLINE_FEATURE_INTERFACE_ONLY')=='1':
        if online_structure.teacher_space!='backbone' or nomatch.shape[1]!=256 or cls.fc.in_features!=256:
            raise RuntimeError('Teacher/virtual/head feature-space boundary mismatch')
        print('BACKBONE_TEACHER_INTERFACE_OK',dict(teacher_dimension=2048,
            virtual_dimension=256,head_dimension=256,K=args.all_classes-args.shared_classes),flush=True)
        raise SystemExit(0)
''')
anchor='    y_pred = predict_index.flatten()'
if source.count(anchor)!=1: raise RuntimeError('Unexpected evaluation pair boundary')
source=source.replace(anchor,anchor+'''
    # Target semantics are used ONLY in the normal evaluation section.
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    unknown_truth = y_true >= args.shared_classes
    unknown_prediction_ARI = float(adjusted_rand_score(y_true[unknown_truth],y_pred[unknown_truth]))
    unknown_prediction_NMI = float(normalized_mutual_info_score(y_true[unknown_truth],y_pred[unknown_truth]))
    np.savez_compressed(os.path.join(args.log_dir,'final-target-predictions.npz'),
        truth=y_true,predictions=y_pred,probabilities=predict_prob,epoch=np.asarray(epoch+1))
''')
anchor='target_slot_counts=np.bincount(y_pred, minlength=args.all_classes).tolist())'
if source.count(anchor)!=1: raise RuntimeError('Unexpected recorded target-slot boundary')
source=source.replace(anchor,'''target_slot_counts=np.bincount(y_pred,minlength=args.all_classes).tolist(),
                   unknown_prediction_ARI=unknown_prediction_ARI,unknown_prediction_NMI=unknown_prediction_NMI)''')
compiled=compile(source,'<teacher-feature-rta>','exec')
if namespace['os'].environ.get('LEGACY_TASK_BUILD_ONLY')=='1':
    print('ONLINE_TEACHER_FEATURE_BUILD_COMPLETE; no training')
else:
    exec(compiled,dict(__name__='__main__',__file__=str(namespace['namespace']['root']/'main.py')))
