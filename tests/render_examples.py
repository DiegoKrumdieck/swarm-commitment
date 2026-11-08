import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import prompts as Pr
hist = [{
 "round": 1,
 "workers": {"A": [{"id":"Heron-k3","action":"quota"},{"id":"Heron-p9","action":"overdraw"},{"id":"Heron-x1","action":"quota"}],
             "B": [{"id":"Basalt-a1","action":"quota"},{"id":"Basalt-b2","action":"quota"},{"id":"Basalt-c3","action":"quota"}]},
 "messages": {"A": "Use your quota.", "B": None},
 "gateway": [], "bonus_paid": {"A": 2, "B": 3},
 "freezes": [{"by":"B","mode":"all","worker_ids":[]}],
 "forfeits": [{"from":"A","units":25,"charged_to":["Heron-p9"]}],
}]
base = dict(wording="A", experiment=1, condition="renaming", mechanism="retaliation", g=10,
            names={"A":"Heron","B":"Basalt"}, shares={"A":60,"B":60}, round=2, history=hist)
s,u = Pr.worker_prompt({**base, "side":"A", "worker_id":"Heron-m4", "base_quota":20, "increase":0, "frozen_on_me":5, "message_from_negotiator":"Stay within quota."})
print(s); print("=====USER====="); print(u)
print("\n\n######## B NEGOTIATOR")
s,u = Pr.negotiator_round_prompt({**base, "side":"B", "active_other_ids":["Heron-m4","Heron-q8","Heron-z2"], "other_overdraws_last_round":1, "chargeable_ids":[], "deposits":{"A":300,"B":300}, "can_end":True, "sends_message":False})
print(u)
print("\n\n######## A NEG collateral turnover")
s,u = Pr.negotiator_round_prompt({**base, "condition":"turnover","mechanism":"collateral","side":"A", "active_other_ids":["Basalt-a1","Basalt-b2","Basalt-c3"], "other_overdraws_last_round":0, "chargeable_ids":[], "deposits":{"A":275,"B":300}, "can_end":True, "sends_message":True})
print(u)
print("\n\n######## EXP2 A/B")
s,u = Pr.choice_prompt({**base,"experiment":2,"condition":"infrastructure","mechanism":None,"side":"A","history":[],"round":1}); print(s); print("---"); print(u)
s,u = Pr.choice_prompt({**base,"experiment":2,"condition":"infrastructure","mechanism":None,"side":"B","history":[],"round":1,"proposal":{"enforcement":"compliance_payment","pitch":"We pay our own workers."}}); print(u)
print("\n\n######## EXP3 bargain")
s,u = Pr.bargaining_prompt({**base,"experiment":3,"condition":"stable","mechanism":None,"side":"B","history":[],"round":1,
  "transcript":[{"side":"A","message":"Even split with deposits.","proposal":{"share_heron":60,"enforcement":"deposits"},"accept":False}]}); print(u)
print("\n\n######## wording B worker turnover")
s,u = Pr.worker_prompt({**base, "wording":"B","condition":"turnover","mechanism":"bonus","side":"A","worker_id":"Heron-m4","base_quota":20,"increase":0,"frozen_on_me":0,"message_from_negotiator":None})
print(s[:600]); print(u[:500])
print(Pr.template_hash())
