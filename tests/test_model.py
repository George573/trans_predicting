from copy import deepcopy
import pytest
import torch
from tram_forecast.config import Config
from tram_forecast.model import ForecastNetwork
from tram_forecast.model.events import EventEncoder

torch.set_num_threads(1)


def test_event_padding_output_and_gradient():
    torch.manual_seed(67)
    original=EventEncoder([12]*5,Config())
    for length in (1,2,13):
        a=deepcopy(original); b=deepcopy(original)
        ids=torch.randint(1,12,(1,length,5))
        padded=torch.zeros(2,length+19,5,dtype=torch.long)
        padded[0,:length]=ids[0];padded[1]=torch.randint(1,12,(length+19,5))
        x=a(ids,torch.tensor([length])); y=b(padded,torch.tensor([length,length+19]))[:1]
        torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)
        x.sum().backward();y.sum().backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)
    assert torch.equal(original(torch.zeros(1,1,5,dtype=torch.long),torch.tensor([0])),torch.zeros(1,32))


def fixture():
    history={'counts':torch.rand(1,1,504)*50,'calendar':torch.rand(1,6,504),
             'event_ids':torch.randint(1,12,(2,9,5)), 'lengths':torch.tensor([9,2]),'hour_indices':torch.tensor([0,503])}
    request={'route_indices':torch.tensor([1]),'calendar':torch.rand(1,4),'lead':torch.tensor([61.])}
    return history,request


def test_network_parameters_forward_backward_and_cache():
    torch.manual_seed(1)
    model=ForecastNetwork([12]*5,20).eval()
    embedding=12*28+80
    assert model.parameter_report()['total']==128704+embedding
    max_model=ForecastNetwork([x+3 for x in Config().category_caps],20)
    assert max_model.parameter_report()['total']==199268
    history,request=fixture()
    out=model(history,request)
    assert out.shape==(1,24) and torch.isfinite(out).all() and (out>=0).all()
    torch.testing.assert_close(out,model.predict_day(model.encode_history(history),request['route_indices'],request['calendar'],request['lead']))
    out.mean().backward()
    for module in [model.events.embeddings,model.raw,model.boarding]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    empty=dict(history,hour_indices=torch.empty(0,dtype=torch.long))
    assert torch.isfinite(model(empty,request)).all()
    with pytest.raises(ValueError,match='unsupported'): model(empty,dict(request,route_indices=torch.tensor([0])))


def test_checkpoint_reload_and_tiny_learning(tmp_path):
    torch.manual_seed(3)
    model=ForecastNetwork([12]*5,1,Config(dropout=0)).train()
    history,request=fixture(); target=torch.full((1,24),2.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    first=None
    for _ in range(12):
        optimizer.zero_grad();loss=(model(history,request)-target).abs().mean()
        if first is None:first=loss.item()
        loss.backward();optimizer.step()
    assert loss.item()<first*0.8
    model.eval();path=tmp_path/'weights.pt';torch.save(model.state_dict(),path)
    restored=ForecastNetwork([12]*5,1,Config(dropout=0)).eval()
    restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(model(history,request),restored(history,request))
