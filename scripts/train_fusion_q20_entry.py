"""Replace only repeated virtual refresh, retaining initial IMP and fixed K."""
from pathlib import Path
import os
import runpy

fusion = runpy.run_path('/content/train_fusion_imp_rta_entry.py',run_name='fusion_builder')
base = Path('/content/train_konly_rta_entry.py').read_text()
namespace = dict(__name__='__main__',__file__='/content/train_konly_rta_entry.py')
exec(compile(base.split('exec(compile(source, ')[0],'<konly-builder>','exec'),namespace)
source = fusion['build_source'](namespace['source'])
anchor = '    nomatch = refresh_virtual(net, args, epoch + 1)'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected virtual update anchor')
source = 'from q20_virtual_control import refresh_q20_virtual\n'+source.replace(anchor,
    '    nomatch = refresh_q20_virtual(net, args, epoch + 1)')
anchor = 'fusion_config_path.write_text(json.dumps(fusion_config, indent=2, allow_nan=False))'
if source.count(anchor) != 1:
    raise RuntimeError('Unexpected virtual configuration anchor')
source = source.replace(anchor,
    "fusion_config.update(design='fusion Q20 virtual control', virtual_policy='initial IMP, then Q20 K-means and Hungarian on same frozen features')\n"+anchor)
compiled = compile(source,'<fusion-q20>','exec')
if os.environ.get('FUSION_BUILD_ONLY') == '1':
    print('Q20_VIRTUAL_BUILD_COMPLETE; no training')
else:
    exec(compiled,namespace)
