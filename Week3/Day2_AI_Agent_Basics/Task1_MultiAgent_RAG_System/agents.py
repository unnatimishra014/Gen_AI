
# agents.py
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Dict, List, Tuple

WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD.findall(text)]

@dataclass
class Document:
    doc_id: str
    text: str

class MiniTfidf:
    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.vocab_idf: Dict[str, float] = {}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self._build()
    
    def _build(self):
        N = len(self.docs)
        df = {}
        tokenized = {}
        for d in self.docs:
            toks = tokenize(d.text)
            tokenized[d.doc_id] = toks
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        for t, c in df.items():
            self.vocab_idf[t] = math.log((N + 1) / (c + 1)) + 1.0
        for d in self.docs:
            tf = {}
            for t in tokenized[d.doc_id]:
                tf[t] = tf.get(t, 0) + 1
            max_tf = max(tf.values()) if tf else 1
            vec = {t: (0.5 + 0.5*tf[t]/max_tf) * self.vocab_idf.get(t, 0.0) for t in tf}
            norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.doc_vectors[d.doc_id] = {t: v/norm for t, v in vec.items()}
    
    def encode_query(self, q: str) -> Dict[str, float]:
        tf = {}
        toks = tokenize(q)
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        if not tf:
            return {}
        max_tf = max(tf.values())
        vec = {t: (0.5 + 0.5*tf[t]/max_tf) * self.vocab_idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t: v/norm for t, v in vec.items()}
    
    def cosine(self, qvec: Dict[str, float], dvec: Dict[str, float]) -> float:
        if not qvec or not dvec:
            return 0.0
        if len(qvec) > len(dvec):
            qvec, dvec = dvec, qvec
        return sum(qvec.get(t,0.0)*dvec.get(t,0.0) for t in qvec.keys())
    
    def search(self, query: str, topk: int = 3) -> List[Tuple[str, float]]:
        qvec = self.encode_query(query)
        scores = []
        for doc_id, dvec in self.doc_vectors.items():
            s = self.cosine(qvec, dvec)
            scores.append((doc_id, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

@dataclass
class AgentResult:
    answer: str
    agent_name: str

class BaseAgent:
    def __init__(self, name: str, index: MiniTfidf, keywords: List[str]):
        self.name = name
        self.index = index
        self.keywords = [t.lower() for t in keywords]

    def run(self, query: str) -> AgentResult:
        hits = self.index.search(query, topk=3)
        best_line, best_score = "", 0.0
        qvec = self.index.encode_query(query)
        for doc_id, _ in hits:
            text = next(d.text for d in self.index.docs if d.doc_id == doc_id)
            # split into lines & ignore empty or header-like
            lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.endswith(":")]
            for ln in lines:
                lvec = self.index.encode_query(ln)
                score = self.index.cosine(qvec, lvec)
                if score > best_score:
                    best_score, best_line = score, ln
        if not best_line:
            best_line = "Sorry, I couldn't find a direct answer."
        return AgentResult(answer=best_line, agent_name=self.name)

class SalaryAgent(BaseAgent): pass
class InsuranceAgent(BaseAgent): pass

class Coordinator:
    def __init__(self, salary_agent: SalaryAgent, insurance_agent: InsuranceAgent):
        self.salary = salary_agent
        self.insurance = insurance_agent
    
    def route(self, query: str) -> AgentResult:
        s_hits = self.salary.index.search(query, topk=3)
        i_hits = self.insurance.index.search(query, topk=3)
        s_score = sum(s for _, s in s_hits) / max(1, len(s_hits))
        i_score = sum(s for _, s in i_hits) / max(1, len(i_hits))
        if i_score > s_score:
            return self.insurance.run(query)
        return self.salary.run(query)

def build_system(salary_docs: List[Document], insurance_docs: List[Document]) -> Coordinator:
    s_idx = MiniTfidf(salary_docs)
    i_idx = MiniTfidf(insurance_docs)
    s_agent = SalaryAgent("Salary Agent", s_idx, ["salary","payslip","ctc","hra","deduction","net","gross","basic"])
    i_agent = InsuranceAgent("Insurance Agent", i_idx, ["insurance","policy","premium","claim","coverage","exclusion","document"])
    return Coordinator(s_agent, i_agent)
