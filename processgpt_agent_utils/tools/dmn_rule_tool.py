from __future__ import annotations
import os
import logging
import xml.etree.ElementTree as ET
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from dotenv import load_dotenv
import openai
import json

from ..utils.database import get_db_client, initialize_db

logger = logging.getLogger(__name__)

# ============================================================================
# 설정 및 초기화
# ============================================================================
load_dotenv()

# ============================================================================
# 스키마 정의
# ============================================================================
class DMNRuleQuerySchema(BaseModel):
    """DMN 규칙 기반 쿼리 추론을 위한 스키마"""
    query: str = Field(..., description="분석할 쿼리 또는 질문")
    context: Optional[str] = Field(None, description="추론 컨텍스트 (선택사항)")

# ============================================================================
# DMN 규칙 도구
# ============================================================================
class DMNRuleTool(BaseTool):
    """DMN(Decision Model and Notation) 규칙 기반 쿼리 추론 도구"""
    name: str = "dmn_rule"
    description: str = (
        "📋 DMN 규칙 기반 쿼리 추론 도구\n\n"
        "주요 기능:\n"
        "• 사용자 소유의 DMN 규칙들을 기반으로 쿼리 분석\n"
        "• 비즈니스 규칙에 따른 자동화된 의사결정 지원\n"
        "• 복잡한 조건부 로직 처리 및 추론\n"
        "• 규칙 기반 질문 답변\n\n"
        "사용 목적:\n"
        "- 비즈니스 규칙에 따른 자동화된 의사결정\n"
        "- 정책 및 규정에 따른 업무 처리\n"
        "- 복잡한 조건부 로직 처리\n"
        "- 규칙 기반 질문 답변\n\n"
        "사용 지침:\n"
        "- 초기화 시 사용자 ID를 소유자로 설정\n"
        "- 쿼리 내용을 분석하여 관련 규칙 적용\n"
        "- 규칙 결과를 바탕으로 답변 제공\n\n"
        "⚠️ 비즈니스 규칙 기반 도구 - 사용자별 규칙 관리"
    )
    args_schema: Type[DMNRuleQuerySchema] = DMNRuleQuerySchema
    _tenant_id: Optional[str] = PrivateAttr()
    _user_id: Optional[str] = PrivateAttr()
    _user_rules: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, tenant_id: str = None, user_id: str = None, **kwargs):
        super().__init__(**kwargs)
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._user_rules = []
        
        # 기존 데이터베이스 클라이언트 초기화
        try:
            initialize_db()
            
            # 초기화 시 사용자 소유의 DMN 규칙들을 미리 조회
            if self._user_id:
                self._load_user_rules()
            
            logger.info("✅ DMNRuleTool 초기화 완료 | tenant_id=%s, user_id=%s, 규칙 개수=%d", 
                       self._tenant_id, self._user_id, len(self._user_rules))
        except Exception as e:
            logger.error("❌ DMNRuleTool 초기화 실패 | tenant_id=%s, user_id=%s err=%s", 
                        self._tenant_id, self._user_id, str(e), exc_info=True)
            raise

    def _load_user_rules(self) -> None:
        """사용자 소유의 DMN 규칙들을 미리 조회"""
        try:
            client = get_db_client()
            response = client.table("proc_def").select("id, name, bpmn, owner, type").eq("owner", self._user_id).eq("type", "dmn").eq("isdeleted", False).eq("tenant_id", self._tenant_id).execute()
            
            self._user_rules = response.data if response.data else []
            logger.info("📋 사용자 DMN 규칙 로드 완료 | user_id=%s, 규칙 개수=%d", self._user_id, len(self._user_rules))
            
        except Exception as e:
            logger.error("❌ 사용자 DMN 규칙 로드 실패 | user_id=%s err=%s", self._user_id, str(e), exc_info=True)
            self._user_rules = []

    def _run(self, query: str, context: Optional[str] = None) -> str:
        """사용자 DMN 규칙들을 기반으로 쿼리 분석 및 추론"""
        logger.info("📋 DMN 규칙 기반 쿼리 분석 시작 | tenant_id=%s, query=%s", self._tenant_id, query)
        
        if not query:
            logger.warning("⚠️ DMN 규칙 처리 실패: 빈 쿼리")
            return "분석할 쿼리를 입력해주세요."

        if not self._user_rules:
            logger.warning("⚠️ 사용자 DMN 규칙이 없습니다 | user_id=%s", self._user_id)
            return f"사용자 '{self._user_id}'의 DMN 규칙이 없습니다."

        try:
            # 쿼리 분석 및 관련 규칙 찾기
            analysis_result = self._analyze_query_with_rules(query, context)
            
            logger.info("✅ DMN 규칙 기반 쿼리 분석 완료 | tenant_id=%s", self._tenant_id)
            return analysis_result

        except Exception as e:
            logger.error("❌ DMN 규칙 기반 쿼리 분석 실패 | tenant_id=%s query=%s err=%s", 
            self._tenant_id, query, str(e), exc_info=True)
            raise

    def _analyze_query_with_rules(self, query: str, context: Optional[str] = None) -> str:
        """사용자 DMN 규칙들을 기반으로 쿼리 분석 및 추론"""
        try:
            # 쿼리에서 키워드 추출
            query_lower = query.lower()
            
            # 관련 규칙 찾기
            relevant_rules = []
            for rule in self._user_rules:
                rule_name = rule.get('name', '').lower()
                if any(keyword in query_lower for keyword in rule_name.split()):
                    relevant_rules.append(rule)
            
            # 규칙 이름으로 직접 매칭되지 않으면 모든 규칙을 고려
            if not relevant_rules:
                relevant_rules = self._user_rules
            
            if not relevant_rules:
                return f"사용자 '{self._user_id}'의 DMN 규칙 중 쿼리와 관련된 규칙을 찾을 수 없습니다."
            
            # 쿼리 분석 및 답변 추론
            answer = self._evaluate_with_rules(query, relevant_rules)
            
            return answer
            
        except Exception as e:
            logger.error("❌ 쿼리 분석 실패 | err=%s", str(e))
            return f"쿼리 분석 중 오류가 발생했습니다: {str(e)}"

    def _evaluate_with_rules(self, query: str, rules: List[Dict[str, Any]]) -> str:
        """DMN 규칙을 기반으로 쿼리 평가 - LLM 분석 사용"""
        try:
            # 모든 규칙을 함께 분석하기 위해 DMN XML들을 수집
            dmn_contexts = []
            for rule in rules:
                rule_name = rule.get('name', '규칙')
                bpmn_xml = rule.get('bpmn')
                
                if bpmn_xml:
                    dmn_structure = self._parse_dmn_to_json(bpmn_xml)
                    if dmn_structure:
                        dmn_contexts.append({
                            'rule_name': rule_name,
                            'dmn_structure': dmn_structure,
                            'bpmn_xml': bpmn_xml
                        })
            
            if not dmn_contexts:
                return f"'{query}'에 대한 DMN 규칙을 찾을 수 없습니다."
            
            return self._ai_inference_with_dmn(dmn_contexts, query)
                
        except Exception as e:
            logger.error("❌ 규칙 평가 실패 | err=%s", str(e))
            return f"규칙 평가 중 오류가 발생했습니다: {str(e)}"

    def _parse_dmn_to_json(self, bpmn_xml: str) -> Optional[Dict[str, Any]]:
        """DMN XML을 JSON 구조로 변환"""
        try:
            root = ET.fromstring(bpmn_xml)
            dmn_ns = {'dmn': 'https://www.omg.org/spec/DMN/20191111/MODEL/'}
            
            decisions = root.findall('.//dmn:decision', dmn_ns)
            
            if not decisions:
                return None
            
            dmn_data = {
                'decisions': []
            }
            
            for decision in decisions:
                decision_name = decision.get('name', 'Decision')
                decision_table = decision.find('.//dmn:decisionTable', dmn_ns)
                
                if decision_table is not None:
                    # Input 분석
                    inputs = decision_table.findall('.//dmn:input', dmn_ns)
                    input_data = []
                    for inp in inputs:
                        input_label = inp.get('label', '')
                        input_expr = inp.find('.//dmn:text', dmn_ns)
                        input_text = input_expr.text if input_expr is not None else ''
                        input_data.append({
                            'label': input_label,
                            'expression': input_text
                        })
                    
                    # Output 분석
                    outputs = decision_table.findall('.//dmn:output', dmn_ns)
                    output_data = []
                    for out in outputs:
                        output_label = out.get('label', '')
                        output_name = out.get('name', '')
                        output_data.append({
                            'label': output_label,
                            'name': output_name
                        })
                    
                    # Rule 분석
                    rules = decision_table.findall('.//dmn:rule', dmn_ns)
                    rule_data = []
                    for rule in rules:
                        input_entries = rule.findall('.//dmn:inputEntry', dmn_ns)
                        output_entries = rule.findall('.//dmn:outputEntry', dmn_ns)
                        
                        conditions = []
                        for entry in input_entries:
                            text_elem = entry.find('.//dmn:text', dmn_ns)
                            if text_elem is not None and text_elem.text and text_elem.text != '-':
                                conditions.append(text_elem.text.strip())
                        
                        results = []
                        for entry in output_entries:
                            text_elem = entry.find('.//dmn:text', dmn_ns)
                            if text_elem is not None and text_elem.text:
                                results.append(text_elem.text.strip())
                        
                        rule_data.append({
                            'conditions': conditions,
                            'results': results
                        })
                    
                    dmn_data['decisions'].append({
                        'name': decision_name,
                        'inputs': input_data,
                        'outputs': output_data,
                        'rules': rule_data
                    })
            
            return dmn_data
            
        except Exception as e:
            logger.error("❌ DMN 파싱 실패 | err=%s", str(e))
            return None
    
    def _ai_inference_with_dmn(self, dmn_contexts: List[Dict[str, Any]], query: str) -> str:
        """AI를 활용한 DMN 규칙 기반 추론 (여러 규칙 컨텍스트 지원)"""
        try:
            # OpenAI API 키 확인
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않음. 기본 분석 모드로 전환")
                return self._fallback_analysis_multi(dmn_contexts, query)
            
            # AI 프롬프트 구성
            prompt = self._build_ai_prompt(dmn_contexts, query)

            # OpenAI API 호출
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """당신은 DMN(Decision Model and Notation) 1.3 규칙 추론 전문가입니다.
주어진 DMN XML 모델과 사용자 질문을 분석하여 비즈니스 의사결정이 어떻게 이루어지는지 설명해야 합니다.

다음 구조로 문서를 작성하세요(입력값이 없어도 규칙 자체 설명은 반드시 포함):

1. **질문 분석 (Question Analysis)**
   - 사용자가 무엇을 묻는지 요약하고 의도를 설명

2. **규칙 요약 (Rule Overview)**
   - 관련 결정 테이블의 입력, 출력, hit policy, 주요 규칙들을 간단히 요약
   - 최소 1개 이상의 규칙 예시를 조건→결과 형태로 제시

3. **규칙 매칭 (Rule Matching)**
   - 마크다운 표: Rule ID | 조건(입력값 기준) | 결과(Output) | 매칭여부(✅/❌)
   - 입력값이 부족해도 표는 채우되 매칭여부는 ❌로 표시하고, 무엇이 부족한지 주석으로 덧붙이기

4. **조건 평가 (Condition Evaluation)**
   - 매칭된(또는 매칭 불가한) 규칙의 조건 충족 여부를 단계별로 설명
   - 사용된 hit policy와 그 의미 설명

5. **최종 결과 (Final Result)**
   - 비즈니스적으로 어떤 결정/등급/혜택이 적용되는지 명확히 제시.
   - 필요시 실제 수치/금액/등급 해석을 포함
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info("✅ AI 추론 완료 | contexts=%d", len(dmn_contexts))
            
            return ai_response
            
        except Exception as e:
            logger.error("❌ AI 추론 실패 | err=%s", str(e))
            # AI 실패 시 폴백 분석 사용 (여러 컨텍스트)
            fallback_answers = []
            for ctx in dmn_contexts:
                rule_name = ctx.get('rule_name', '규칙')
                dmn_structure = ctx.get('dmn_structure')
                if dmn_structure:
                    fallback_answers.append(self._fallback_analysis(dmn_structure, rule_name, query))
            return "\n\n".join(ans for ans in fallback_answers if ans)

    def _build_ai_prompt(self, dmn_contexts: List[Dict[str, Any]], query: str) -> str:
        """AI 추론을 위한 프롬프트 구성"""
        prompt_parts = []
        
        # DMN 규칙들 설명
        for i, ctx in enumerate(dmn_contexts, 1):
            rule_name = ctx['rule_name']
            dmn_structure = ctx['dmn_structure']
            
            prompt_parts.append(f"=== DMN 규칙 {i}: {rule_name} ===\n")
            prompt_parts.append(json.dumps(dmn_structure, ensure_ascii=False, indent=2))
            prompt_parts.append("\n")
        
        # 사용자 쿼리
        prompt_parts.append(f"\n사용자 쿼리: \"{query}\"\n")
        
        # 지시사항
        prompt_parts.append("""
위 DMN 규칙들을 분석하여 다음을 수행해주세요.

1. **질문 분석**: 사용자가 무엇을 묻는지 요약하고 의도를 설명
2. **규칙 요약**: 입력/출력, hit policy, 대표 규칙 예시(조건→결과)
3. **규칙 매칭**: 표 (Rule ID | 조건 | 결과 | 매칭여부). 입력값이 없으면 ❌과 부족한 항목 주석
4. **조건 평가**: 충족/미충족 사유와 hit policy 설명
5. **최종 결과**: 비즈니스적으로 어떤 결정/등급/혜택이 적용되는지 명확히 제시.

답변은 한국어 마크다운으로 작성하고, 이모지(✅, 💡, 🔍 등)를 사용해 가독성을 높여주세요.
""")
        
        return "".join(prompt_parts)

    def _fallback_analysis(self, dmn_structure: Dict[str, Any], rule_name: str, query: str) -> str:
        """AI 실패 시 사용할 기본 분석"""
        try:
            answers = []
            
            for decision in dmn_structure['decisions']:
                decision_name = decision['name']
                inputs = decision['inputs']
                outputs = decision['outputs']
                rules = decision['rules']
                
                # 입력 파라미터 설명
                input_descriptions = []
                for inp in inputs:
                    if inp['expression']:
                        input_descriptions.append(inp['expression'])
                
                # 규칙 설명
                rule_descriptions = []
                for rule in rules[:5]:  # 처음 5개 규칙만
                    if rule['conditions'] and rule['results']:
                        conditions_str = ' AND '.join(rule['conditions'])
                        results_str = ', '.join(rule['results'])
                        rule_descriptions.append(f"  • {conditions_str} → {results_str}")
                
                # 답변 생성
                if '어떻게' in query.lower() or 'how' in query.lower():
                    # 방법 설명
                    answer = f"{rule_name}의 {decision_name}은 다음과 같이 작동합니다:\n"
                    if input_descriptions:
                        answer += f"- 입력: {', '.join(input_descriptions)}\n"
                    if rule_descriptions:
                        answer += "- 규칙 예시:\n" + "\n".join(rule_descriptions)
                    answers.append(answer)
                
                else:
                    # 평가/결정 답변
                    answer = f"{rule_name}의 {decision_name}에 따르면:\n"
                    if input_descriptions:
                        answer += f"- 평가 기준: {', '.join(input_descriptions)}\n"
                    if rule_descriptions:
                        answer += "- 규칙:\n" + "\n".join(rule_descriptions)
                    answers.append(answer)
            
            return "\n\n".join(answers) if answers else f"{rule_name}: 쿼리에 대한 답변을 생성할 수 없습니다."
            
        except Exception as e:
            logger.error("❌ 폴백 분석 실패 | err=%s", str(e))
            return f"{rule_name}: 분석 실패 - {str(e)}"
