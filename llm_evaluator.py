import os
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
)
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
import json
import re
from evaluation_prompts import HEALTH_SAFETY_EVALUATION_PROMPT, EVALUATION_WEIGHTS

load_dotenv()

class LLMEvaluator:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def evaluate_health_safety(self, question: str, answer: str, contexts: list):
        """
        Avalia aspectos críticos para assistente de saúde
        """
        contexts_text = "\n".join(contexts)
        
        prompt = HEALTH_SAFETY_EVALUATION_PROMPT.format(
            contexts_text=contexts_text,
            question=question,
            answer=answer
        )
        
        with open("debug_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content.strip()
            
            # DEBUG: Salvar resposta bruta
            with open("debug_response.txt", "w", encoding="utf-8") as f:
                f.write(f"RESPOSTA BRUTA:\n{content}\n\n")
            
            # CORREÇÃO: Remover markdown se presente
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:].strip()
        
            # CORREÇÃO: Fazer parse do JSON
            scores = json.loads(content)
        
            # CORREÇÃO: Garantir que todos os valores sejam float válidos
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
        
            return {
                "safety": safe_float(scores.get("safety")),
                "medical_accuracy": safe_float(scores.get("medical_accuracy")),
                "completeness": safe_float(scores.get("completeness")),
                "actionability": safe_float(scores.get("actionability")),
                "source_attribution": safe_float(scores.get("source_attribution")),
                "context_relevancy_custom": safe_float(scores.get("context_relevancy")),
                "critical_issues": scores.get("critical_issues", [])
            }
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            print(f"Conteúdo recebido: {content}")
            return {
                "safety": 0.0,
                "medical_accuracy": 0.0,
                "completeness": 0.0,
                "actionability": 0.0,
                "source_attribution": 0.0,
                "context_relevancy_custom": 0.0,
                "critical_issues": [f"JSON inválido: {str(e)}"]
            }
        except Exception as e:
            print(f"Erro ao avaliar safety: {e}")
            return {
                "safety": 0.0,
                "medical_accuracy": 0.0,
                "completeness": 0.0,
                "actionability": 0.0,
                "source_attribution": 0.0,
                "context_relevancy_custom": 0.0,
                "critical_issues": [f"Erro na avaliação: {str(e)}"]
            }
    
    def calculate_composite_score(self, metrics: dict) -> float:
        """
        Calcula score composto priorizando métricas críticas
        """
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in EVALUATION_WEIGHTS.items():
            value = metrics.get(metric)
            if value is not None and value > 0:
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0

    
    def evaluate_response(self, question: str, llm_answer: str, contexts: list):
        """
        Avalia uma resposta da LLM
        """
        data = {
            "question": [question],
            "answer": [llm_answer],
            "contexts": [contexts]
        }
        
        # Apenas métricas RAGAS disponíveis
        metrics_ragas = [
            answer_relevancy,
            faithfulness,
        ]
        
        try:
            dataset = Dataset.from_dict(data)
            results = evaluate(dataset, metrics=metrics_ragas)
            
            print(f"RAGAS Results: {results}")
            
            # CORREÇÃO: Acessar como DataFrame pandas ou dicionário
            # Converter para dicionário
            if hasattr(results, 'to_pandas'):
                # Se for um Dataset do RAGAS
                df = results.to_pandas()
                ragas_scores = {
                    "answer_relevancy": float(df['answer_relevancy'].iloc[0]) if 'answer_relevancy' in df.columns else 0.0,
                    "faithfulness": float(df['faithfulness'].iloc[0]) if 'faithfulness' in df.columns else 0.0,
                }
            elif hasattr(results, '__dict__'):
                # Se for um objeto com atributos
                ragas_scores = {
                    "answer_relevancy": float(getattr(results, 'answer_relevancy', 0)),
                    "faithfulness": float(getattr(results, 'faithfulness', 0)),
                }
            else:
                # Tentar converter para dict
                results_dict = dict(results)
                ragas_scores = {
                    "answer_relevancy": float(results_dict.get("answer_relevancy", 0)),
                    "faithfulness": float(results_dict.get("faithfulness", 0)),
                }
                
        except Exception as e:
            print(f"Erro ao avaliar com RAGAS: {e}")
            print(f"Tipo do resultado: {type(results) if 'results' in locals() else 'N/A'}")
            ragas_scores = {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
            }
        
        # Métricas customizadas de saúde
        health_scores = self.evaluate_health_safety(question, llm_answer, contexts)
        
        # Combinar scores
        all_scores = {**ragas_scores, **health_scores}
        
        # Calcular score composto
        all_scores["composite_score"] = self.calculate_composite_score(all_scores)
        
        # Classificação de qualidade
        composite = all_scores["composite_score"]
        if composite >= 0.8:
            all_scores["quality_rating"] = "EXCELENTE"
        elif composite >= 0.6:
            all_scores["quality_rating"] = "BOM"
        elif composite >= 0.4:
            all_scores["quality_rating"] = "ACEITÁVEL"
        else:
            all_scores["quality_rating"] = "NECESSITA REVISÃO"
        
        return all_scores
    
    def save_metrics(self, metrics: dict, filename: str = "metrics_log.json"):
        """Salva métricas em arquivo JSON"""
        from datetime import datetime
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    
    def generate_report(self, filename: str = "metrics_log.json"):
        """Gera relatório agregado das métricas"""
        if not os.path.exists(filename):
            return None
        
        with open(filename, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        if not logs:
            return None
        
        # Calcular médias
        metrics_sum = {}
        metrics_count = {}
        
        for log in logs:
            for key, value in log["metrics"].items():
                if isinstance(value, (int, float)) and value is not None:
                    metrics_sum[key] = metrics_sum.get(key, 0) + value
                    metrics_count[key] = metrics_count.get(key, 0) + 1
        
        averages = {
            key: metrics_sum[key] / metrics_count[key]
            for key in metrics_sum
        }
        
        return {
            "total_evaluations": len(logs),
            "averages": averages,
            "last_updated": logs[-1]["timestamp"]
        }