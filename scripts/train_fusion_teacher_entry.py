"""Fusion K7 with geometric unknown pseudo-labels; separate method entry."""
from pathlib import Path
import os
import runpy


def build_teacher(source):
    def replace_once(old,new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Unexpected teacher anchor: '+old[:90])
        source = source.replace(old,new)
    replace_once('\nepoch = 0\n','\nunknown_teacher = UnknownPrototypeTeacher(args.shared_classes, momentum=0.9)\nepoch = 0\n')
    replace_once('                _, pseudo_index = predict_prob_otherep[:,args.shared_classes:].max(1)\n'
                 '                pseudo_index=pseudo_index + args.shared_classes',
                 '                pseudo_index = unknown_teacher.assign(feature_otherep, logits_otherep)')
    replace_once('        init_unk_weight = np.stack(init_unk_weight,axis=0)',
                 '        init_unk_weight = np.stack(init_unk_weight,axis=0)\n'
                 '        unknown_teacher.initialize(torch.from_numpy(init_unk_weight).to(next(net.parameters())), args.log_dir)')
    replace_once('    nomatch = refresh_virtual(net, args, epoch + 1)',
                 '    unknown_teacher.save(epoch + 1)\n    nomatch = refresh_virtual(net, args, epoch + 1)')
    replace_once('fusion_config_path.write_text(json.dumps(fusion_config, indent=2, allow_nan=False))',
                 "fusion_config.update(design='fusion geometric unknown teacher v1', unknown_pseudo_label='nearest EMA prototype', teacher_momentum=0.9)\n"
                 'fusion_config_path.write_text(json.dumps(fusion_config, indent=2, allow_nan=False))')
    return 'from unknown_prototype_teacher import UnknownPrototypeTeacher\n'+source


fusion = runpy.run_path('/content/train_fusion_imp_rta_entry.py',run_name='fusion_builder')
base = Path('/content/train_konly_rta_entry.py').read_text()
namespace = dict(__name__='__main__',__file__='/content/train_konly_rta_entry.py')
exec(compile(base.split('exec(compile(source, ')[0],'<konly-builder>','exec'),namespace)
source = build_teacher(fusion['build_source'](namespace['source']))
compiled = compile(source,'<fusion-teacher>','exec')
if os.environ.get('FUSION_BUILD_ONLY') == '1':
    print('UNKNOWN_TEACHER_BUILD_COMPLETE; no training')
else:
    exec(compiled,namespace)
