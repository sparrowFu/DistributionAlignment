import json,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()))
import scripts.ablation_study.mechanism_diagnostics as diag
from losses.mcdisp_align_losses_kl import MCDispAlignKLLoss
observations=[]
def stable_grad_norm(term,inputs):
    if not term.requires_grad:return None
    inputs=[t for t in inputs if t.requires_grad]
    if not inputs:return None
    grads=torch.autograd.grad(term,inputs,retain_graph=True,allow_unused=True)
    if all(g is None for g in grads):return None
    squares=0.0;bad=0;maximum=0.0
    for g in grads:
        if g is None:continue
        gd=g.detach()
        bad+=int((~torch.isfinite(gd)).sum().item())
        maximum=max(maximum,float(gd.abs().max().item()))
        squares+=float(gd.double().square().sum().item())
    norm=math.sqrt(squares)
    observations.append({'nonfinite_gradient_elements':bad,'max_abs_element':maximum,'norm_float64_accumulation':norm})
    return norm
diag._grad_norm=stable_grad_norm
tag,checkpoint,epoch=sys.argv[1],Path(sys.argv[2]),int(sys.argv[3])
crit=MCDispAlignKLLoss(lambda_kl=1,lambda_cover_pos=.5,lambda_cov=.01,lambda_reg=.01,tau=.07,m_pos=1,target_var=.04)
result=diag.term_gradient_probe(checkpoint,Path('outputs/rebuttal_exp3_20261113/manifest_original_val_seed42_diagnostic1000.json'),'cuda',8,4,epoch,3326,10,False,3327,crit)
out={'checkpoint':str(checkpoint),'probe':'same float32 autograd gradients; norm accumulation promoted to float64 only; no parameter updates','term_gradients':result,'gradient_observations':observations,'total_nonfinite_gradient_elements':sum(o['nonfinite_gradient_elements'] for o in observations)}
p=Path('outputs/rebuttal_exp3_20261113')/(tag+'_stable_gradients.json')
p.write_text(json.dumps(out,indent=2))
print(json.dumps({'output':str(p),'total_nonfinite_gradient_elements':out['total_nonfinite_gradient_elements'],'cov':result['cov']},indent=2),flush=True)
