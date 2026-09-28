"""
final_evaluate_LLMs_models.py

Multilingual LLM evaluation using Free Groq Models with Rate-Limit Awareness.
Evaluates 60 articles (20 AR, 20 EN, 20 FR across 5 balanced categories).

Metrics Evaluated:
1. Language Accuracy (via langdetect)
2. Category Classification Accuracy
3. Summary Quality (via LLM-as-a-Judge)
4. Error Rate & Latency
5. Effective Quality = Quality x Faithfulness / 5
   (penalizes fluent-but-hallucinating models; faithfulness measures whether
   the summary invents facts, quality measures clarity/coherence/readability
   — a model can score high on one and low on the other, so neither alone
   is a safe "best model" signal)
"""

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import httpx
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langdetect import DetectorFactory, detect


DetectorFactory.seed = 0

# ============================================================
# API CONFIGURATION
# ============================================================

env_path = Path(__file__).with_name(".env")
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

_no_proxy_client = httpx.Client(
    trust_env=False,
    timeout=60.0,
)

# ============================================================
# CANDIDATE MODELS & JUDGE MODEL
# ============================================================

CANDIDATE_MODELS = {}
JUDGE_MODEL = None

if GROQ_API_KEY:
    CANDIDATE_MODELS["Qwen-3.8-27B"] = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        api_key=GROQ_API_KEY,
        http_client=_no_proxy_client,
    )
    CANDIDATE_MODELS["ALLAM-2-7B"] = ChatGroq(
        model="allam-2-7b",
        temperature=0,
        api_key=GROQ_API_KEY,
        http_client=_no_proxy_client,
    )

    # JUDGE_MODEL = ChatOpenAI(
    #     model="openai/gpt-4o-mini",
    #     api_key=OPENROUTER_API_KEY,
    #     base_url="https://openrouter.ai/api/v1",
    #     temperature=0,
    #     max_tokens=400,
    #     request_timeout=30.0
    # )
    JUDGE_MODEL = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=GROQ_API_KEY,
        http_client=_no_proxy_client,
    )

# ============================================================
# DATASET
# ============================================================

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "fr": "French",
}


def get_extended_dummy_chunks():
    """Generates the multilingual news article dataset: 20 AR, 20 EN, 20 FR
    (4 articles x 5 balanced categories per language: Health, Sports,
    Economy, Technology, Culture). Unchanged from the source dataset —
    only the scoring/selection logic elsewhere in this file was modified."""
    return [
        # ============================================================
        # --- ARABIC (20 CHUNKS - 4 PER CATEGORY) ---
        # ============================================================
        # Health (4)
        Document(
            page_content="أعلنت وزارة الصحة والسكان عن إطلاق المرحلة الثانية من الحملة القومية الموسعة للتطعيم والوقاية من الأمراض المعدية للأطفال دون سن الخامسة في جميع المحافظات. وتستهدف الحملة الوصول إلى أكثر من عشرة ملايين طفل من خلال فرق طبية مدربة تتنقل بين المنازل والوحدات الصحية الريفية، مع توفير كافة اللقاحات بالمجان لضمان تعزيز المناعة المجتمعية والحد من انتشار الأوبئة الموسمية.",
            metadata={"doc_id": "doc_ar_health_1", "language": "ar", "expected_category": "Health"}
        ),
        Document(
            page_content="أظهرت دراسة طبية حديثة أجراها مركز البحوث القومي أن ممارسة النشاط البدني المتوسط مثل المشي السريع لمدة ثلاثين دقيقة يومياً تؤدي إلى تحسين وظائف الأوعية الدموية وتقليل خطر الإصابة بأمراض القلب والشرايين بنسبة تتجاوز الأربعين بالمئة. وأكد الباحثون أن الاتساق في ممارسة الرياضة بات ينظر إليه كعامل رئيسي للوقاية من مرض السكري من النوع الثاني والسيطرة على مستويات الضغط.",
            metadata={"doc_id": "doc_ar_health_2", "language": "ar", "expected_category": "Health"}
        ),
        Document(
            page_content="افتتحت المستشفيات الجامعية قسماً متطوراً لعلاج أورام الأطفال مجهزاً بأحدث أجهزة العلاج الإشعاعي والتشخيص المتقدم بالرنين المغناطيسي. ويهدف هذا القسم الجديد إلى توفير بروتوكولات علاجية مجانية وشاملة طبقاً للمواصفات العالمية، مما يساهم بشكل كبير في تقليل قوائم الانتظار وتقديم الدعم النفسي والطبي اللازم للأطفال وأسرهم طوال فترة الاستشفاء.",
            metadata={"doc_id": "doc_ar_health_3", "language": "ar", "expected_category": "Health"}
        ),
        Document(
            page_content="حذرت منظمة الصحة العالمية في تقريرها الإقليمي الأخير من الارتفاع المقلق في معدلات السمنة والاضطرابات الغذائية بين المراهقين والأطفال نتيجة الممارسات الحياتية الخاطئة والاعتماد المفرط على الوجبات السريعة والمشروبات الغازية. ودعت المنظمة الحكومات إلى فرض قيود صرامة على الإعلانات التجارية الموجهة للأطفال وتشجيع المدارس على إدراج برامج للتوعية بالغذائي الصحي.",
            metadata={"doc_id": "doc_ar_health_4", "language": "ar", "expected_category": "Health"}
        ),

        # Sports (4)
        Document(
            page_content="حقق المنتخب المصري لكرة القدم فوزاً ثميناً على نظيره التونسي بهدفين مقابل هدف واحد في المباراة الودية الدولية التي أقيمت على استاد القاهرة الدولي وسط حضور جماهيري غفير. وسجل هدف الفوز مهاجم الفريق في الدقائق الأخيرة من شوط المباراة الثاني بعد هجمة مرتدة سريعة، مما منح الفريق دفعة معنوية قوية قبل انطلاق التصفيات المؤهلة لكأس الأمم الأفريقية.",
            metadata={"doc_id": "doc_ar_sports_1", "language": "ar", "expected_category": "Sports"}
        ),
        Document(
            page_content="حصد السباح المصري الميدالية الذهبية في سباق مئة متر فراشة ضمن منافسات بطولة العالم للسباحة المقامة حالياً بمدينة برلين الألمانية. واستطاع السباح كسر الرقم القياسي السابق المعتمد في البطولة بفارق أجزاء من الثانية، ليعزز مكانته كأحد أبرز الرياضيين العرب المتأهلين للمشاركة في دورة الألعاب الأولمبية القادمة.",
            metadata={"doc_id": "doc_ar_sports_2", "language": "ar", "expected_category": "Sports"}
        ),
        Document(
            page_content="أعلن مجلس إدارة النادي الأهلي رسمياً عن إنهاء إجراءات التعاقد مع صانع ألعاب جديد قادماً من الدوري البرازيلي في صفقة تمتد لأربعة مواسم قادمة. وصرح المدير الفني للنادي بأن اللاعب يمتلك خبرات عريضة ومهارات فردية عالية ستساهم في دعم خط هجوم الفريق في المنافسات القارية والمحلية القادمة.",
            metadata={"doc_id": "doc_ar_sports_3", "language": "ar", "expected_category": "Sports"}
        ),
        Document(
            page_content="تأهل منتخب مصر لكرة اليد إلى الدور نصف النهائي لبطولة القارة بعد إقصاء نظيره الجزائري بنتيجة ثلاثين مقابل خمسة وعشرين في مباراة شهدت ندية كبيرة وسيطرة تكتيكية. واستطاع حارس مرمى المنتخب التصدي للعديد من الكرات الحاسمة في الدقائق الأخيرة ليؤمن فوز الفريق وبطاقة العبور للمباراة النهائية.",
            metadata={"doc_id": "doc_ar_sports_4", "language": "ar", "expected_category": "Sports"}
        ),

        # Economy (4)
        Document(
            page_content="سجلت مؤشرات البورصة المصرية ارتفاعاً جماعياً كبيراً في ختام تعاملات اليوم، مدفوعة بمشتريات قوية من قبل المؤسسات المالية والمستثمرين الأجانب. وقاد قطاع العقارات والخدمات المالية الارتفاعات بعد الإعلان عن مشروعات استثمارية جديدة وتحسن معدلات السيولة، مما ساهم في زيادة رأس المال السوقي للأسهم المقيدة بقيمة تجاوزت عشرين مليار جنيه.",
            metadata={"doc_id": "doc_ar_economy_1", "language": "ar", "expected_category": "Economy"}
        ),
        Document(
            page_content="قررت لجنة السياسة النقدية بالبنك المركزي تثبيت أسعار الفائدة على الإيداع والإقراض لليلة واحدة عند مستوياتها الحالية، مشيرة إلى أن القرار يأتي بهدف الحفاظ على الاستقرار المالي ومتابعة تقلبات معدلات التضخم العالمية. وأكد البنك في بيانه استمراره في مراقبة كافة المؤشرات الاقتصادية والتدخل بالتعديل المناسب لضمان خفض معدلات التضخم وتوفير بيئة جاذبة للاستثمار.",
            metadata={"doc_id": "doc_ar_economy_2", "language": "ar", "expected_category": "Economy"}
        ),
        Document(
            page_content="وقعت وزارة المالية اتفاقية تمويل جديدة مع البنك الدولي بقيمة مليار دولار مخصصة لدعم مشروعات البنية التحتية الشاملة وتطوير الشبكة الكهربائية والمرافق العامة. وتأتي هذه الخطوة في إطار خطة الحكومة للتوسع في الاعتماد على الطاقة النظيفة وتشجيع الاستثمارات الخضراء بالشراكة مع القطاع الخاص لتحقيق التنمية المستدامة.",
            metadata={"doc_id": "doc_ar_economy_3", "language": "ar", "expected_category": "Economy"}
        ),
        Document(
            page_content="أظهرت بيانات وزارة الزراعة أن الصادرات الزراعية المصرية حققت نمواً بنسبة خمسة عشر بالمئة خلال النصف الأول من العام الحالي مقارنة بالفترة نفسها من العام الماضي. وجاء هذا النمو نتيجة فتح أسواق عالمية جديدة والتوسع في استصلاح الأراضي الزراعية وتطبيق معايير الجودة الشاملة لضمان منافسة المحاصيل المحلية في الأسواق الأوربية والأسيوية.",
            metadata={"doc_id": "doc_ar_economy_4", "language": "ar", "expected_category": "Economy"}
        ),

        # Technology (4)
        Document(
            page_content="أطلقت شركة ناشئة متخصصة في الحلول البرمجية منصة ذكاء اصطناعي تعتمد على خوارزميات التعلم العميق لترشيد استهلاك الطاقة داخل المباني والمصانع الذكية. وتستطيع المنصة تحليل البيانات الآتية من المستشعرات بدقة عالية والتنبؤ بأوقات الذروة لخف الأحمال الكهربائية تلقائياً، مما يساعد الشركات على تخفيض تكاليف التشبيل والانبعاثات الكربونية بنسبة كبيرة.",
            metadata={"doc_id": "doc_ar_tech_1", "language": "ar", "expected_category": "Technology"}
        ),
        Document(
            page_content="نجح فريق بحثي من المهندسين الشباب في تصميم كبسولة فضائية مصغرة لاختبار كفاءة شبكات الاتصالات اللاسلكية في المدار الأرضي المنخفض. وتم إطلاق الكبسولة بنجاح على متن صاروخ تجاري لإجراء تجارب تتعلق بتقنيات إنترنت الأشياء والاتصالات الفضائية القادرة على العمل في الظروف المناخية القاسية والبيئات النائية.",
            metadata={"doc_id": "doc_ar_tech_2", "language": "ar", "expected_category": "Technology"}
        ),
        Document(
            page_content="أعلنت الجهاز القومي لتنظيم الاتصالات عن البدء في إتاحة نطاقات التردد الخاصة بتقديم خدمات الجيل الخامس لشركات المحمول العاملة في السوق المحلي. وتتيح هذه التقنية نقل البيانات بسرعة فائقة وزمن استجابة منخفض جداً، مما يمهد الطريق لتطبيقات الرعاية الصحية عن بعد، والمواصلات الذكية، وإدارة الأجهزة الصناعية المستقلة.",
            metadata={"doc_id": "doc_ar_tech_3", "language": "ar", "expected_category": "Technology"}
        ),
        Document(
            page_content="كشفت شركة عالمية للتقنية عن معالج كمبيوتر جديد يعتمد على بنية الحوسبة الكمومية لتسريع معالجة البيانات المعقدة والمحاكاة الكيميائية. ويمتاز المعالج بقدرته على إجراء ملايين الحسابات في أجزاء من الثانية مقارنة بالحواسيب الفائقة التقليدية، مما يشكل طفرة حقيقية في مجالات اكتشاف الأدوية وتحليل التشفير الإلكتروني.",
            metadata={"doc_id": "doc_ar_tech_4", "language": "ar", "expected_category": "Technology"}
        ),

        # Culture (4)
        Document(
            page_content="شهد معرض القاهرة الدولي للكتاب افتتاح دورته الجديدة بمشاركة مئات دور النشر العربية والأجنبية التي عرضت أحدث الإصدارات الأدبية والفكرية. وتضمن البرنامج الثقافي للمعرض ندوات نقدية وأمسيات شعرية شارك فيها كبار الكاتب والمفكرين، إلى جانب تخصص أجنحة خاصة للأطفال وورش عمل لتعزيز شغف القراءة والتأليف لدى الشباب.",
            metadata={"doc_id": "doc_ar_culture_1", "language": "ar", "expected_category": "Culture"}
        ),
        Document(
            page_content="فاز الفيلم الروائي الطويل بالجائزة الكبرى في مهرجان الإسكندرية السينمائي الدولي في دورته الحالية بعد منافسة قوية مع سينمائيين من دول البحر المتوسط. وأشادت لجنة التحكيم بالحبكة الدرامية والرؤية الإخراجية المتميزة التي عالجت القضايا الاجتماعية بطرح إنساني عميق وأداء تمثيلي حظي بإشادة واسعة من النقاد.",
            metadata={"doc_id": "doc_ar_culture_2", "language": "ar", "expected_category": "Culture"}
        ),
        Document(
            page_content="نظمت وزارة الثقافة احتفالية موسيقية ضخمة على مسرح دار الأوبرا لإحياء التراث الغنائي وتكريم رواد الموسيقى العربية الأصيلة. وقدمت الفرقة القومية للموسيقى أوركسترا حيًا تضمن روائع الموشحات والأغاني التراثية التي أعيد توزيعها بأسلوب عصري يمزج بين الأصالة والروح التجديدية للموسيقى الكلاسيكية.",
            metadata={"doc_id": "doc_ar_culture_3", "language": "ar", "expected_category": "Culture"}
        ),
        Document(
            page_content="نجحت البعثة الأثرية المصرية العاملة في منطقة الأقصر في الكشف عن مقبرة فرعونية جديدة تعود إلى عصر الأسرة الثامنة عشرة وتحتوي على جداريات نادرة ملونة بحالة جيدة. وأوضح الأثريون أن المقبرة تضم عدداً من التماثيل والأواني الكانوبية والمقتنيات الجنائزية التي تسلط الضوء على الحياة اليومية والطقوس الدينية في تلك الفترة التاريخية.",
            metadata={"doc_id": "doc_ar_culture_4", "language": "ar", "expected_category": "Culture"}
        ),

        # ============================================================
        # --- ENGLISH (20 CHUNKS - 4 PER CATEGORY) ---
        # ============================================================
        # Health (4)
        Document(
            page_content="The Ministry of Health announced the launch of a new comprehensive nationwide vaccination campaign targeting children under five years old across all governorates. The initiative aims to deploy mobile health units to remote rural areas, providing free vaccines to prevent infectious diseases, strengthen community immunity, and reduce seasonal pediatric hospitalizations significantly.",
            metadata={"doc_id": "doc_en_health_1", "language": "en", "expected_category": "Health"}
        ),
        Document(
            page_content="A recent medical study published by researchers indicates that engaging in moderate physical exercises like brisk walking for thirty minutes every day drastically lowers the risk of developing cardiovascular diseases by up to forty percent. Researchers stressed that regular lifestyle modifications serve as a crucial preventative measure against type-2 diabetes and hypertension.",
            metadata={"doc_id": "doc_en_health_2", "language": "en", "expected_category": "Health"}
        ),
        Document(
            page_content="University hospitals inaugurate a state-of-the-art pediatric oncology department equipped with modern radiotherapy tools and advanced diagnostic MRI scanners. This facility provides free, world-class cancer treatment protocols to underprivileged patients, aiming to reduce long waiting lists and provide psychological support to families during recovery.",
            metadata={"doc_id": "doc_en_health_3", "language": "en", "expected_category": "Health"}
        ),
        Document(
            page_content="The World Health Organization warned in its latest regional report about the alarming rise in obesity and dietary disorders among adolescents due to sedentary lifestyles and reliance on fast food. The report urges policymakers to enforce stricter nutritional labeling laws and support educational health campaigns within public schools.",
            metadata={"doc_id": "doc_en_health_4", "language": "en", "expected_category": "Health"}
        ),

        # Sports (4)
        Document(
            page_content="The Egyptian national football team secured a thrilling 2-1 victory over Tunisia in an international friendly match played at Cairo International Stadium. The winning goal came in the final minutes of the second half following a swift counter-attack, boosting team confidence ahead of the upcoming continental championship qualifiers.",
            metadata={"doc_id": "doc_en_sports_1", "language": "en", "expected_category": "Sports"}
        ),
        Document(
            page_content="An Egyptian swimmer clinched the gold medal in the 100-meter butterfly event at the World Aquatics Championships held in Berlin. The athlete broke the previous championship record by fraction of a second, solidifying his position as one of the top contenders for the upcoming Olympic Games.",
            metadata={"doc_id": "doc_en_sports_2", "language": "en", "expected_category": "Sports"}
        ),
        Document(
            page_content="Al Ahly Club officially announced the signing of a promising Brazilian playmaker on a four-year contract. The head coach expressed enthusiasm regarding the transfer, highlighting the player's tactical versatility and attacking capabilities, which are expected to strengthen the squad in upcoming domestic and continental leagues.",
            metadata={"doc_id": "doc_en_sports_3", "language": "en", "expected_category": "Sports"}
        ),
        Document(
            page_content="The national handball squad advanced to the tournament semifinals after defeating Algeria 30-25 in a fiercely contested match. Spectacular saves by the goalkeeper during the final quarter proved decisive in preserving the lead and securing a spot in the championship final.",
            metadata={"doc_id": "doc_en_sports_4", "language": "en", "expected_category": "Sports"}
        ),

        # Economy (4)
        Document(
            page_content="Stock market indices posted broad gains at the close of trading today, driven by strong buying interest from institutional foreign investors. The real estate and banking sectors led the rally following announcements of new foreign investment partnerships and improved market liquidity.",
            metadata={"doc_id": "doc_en_economy_1", "language": "en", "expected_category": "Economy"}
        ),
        Document(
            page_content="The Central Bank's Monetary Policy Committee decided to keep key benchmark interest rates unchanged to maintain macroeconomic stability and curb inflation. Officials noted that monetary policy will remain vigilant, adjusting policy levers as necessary to support sustainable economic growth and stabilize consumer prices.",
            metadata={"doc_id": "doc_en_economy_2", "language": "en", "expected_category": "Economy"}
        ),
        Document(
            page_content="The Ministry of Finance signed a one billion dollar financing agreement with the World Bank aimed at modernizing national infrastructure and renewable energy grids. The initiative supports public-private partnerships to expand solar and wind power installations while reducing reliance on fossil fuels.",
            metadata={"doc_id": "doc_en_economy_3", "language": "en", "expected_category": "Economy"}
        ),
        Document(
            page_content="Agricultural exports grew by fifteen percent during the first half of the year compared to the same period last year, according to official trade data. Expanding export destinations, land reclamation projects, and strict quality control protocols contributed to higher international demand for local produce.",
            metadata={"doc_id": "doc_en_economy_4", "language": "en", "expected_category": "Economy"}
        ),

        # Technology (4)
        Document(
            page_content="A tech startup released an artificial intelligence platform utilizing deep learning algorithms to optimize energy consumption in smart buildings. By analyzing real-time sensor data, the software predicts peak demand loads and adjusts HVAC systems automatically, reducing carbon emissions and utility expenses.",
            metadata={"doc_id": "doc_en_tech_1", "language": "en", "expected_category": "Technology"}
        ),
        Document(
            page_content="Aerospace engineers successfully launched a miniature satellite prototype designed to evaluate low-Earth orbit satellite communications. The mission tests IoT connectivity performance and data transmission reliability under extreme space radiation and atmospheric conditions.",
            metadata={"doc_id": "doc_en_tech_2", "language": "en", "expected_category": "Technology"}
        ),
        Document(
            page_content="Telecommunication regulators began allocating 5G frequency spectrum licenses to mobile operators across the country. The ultra-fast bandwidth and low latency offered by 5G networks are expected to accelerate tele-health services, autonomous transport, and industrial automation across major cities.",
            metadata={"doc_id": "doc_en_tech_3", "language": "en", "expected_category": "Technology"}
        ),
        Document(
            page_content="A global technology firm unveiled a next-generation quantum computing processor designed for complex data simulation and molecular modeling. The processor performs computations exponentially faster than traditional supercomputers, paving the way for breakthroughs in material science and drug discovery.",
            metadata={"doc_id": "doc_en_tech_4", "language": "en", "expected_category": "Technology"}
        ),

        # Culture (4)
        Document(
            page_content="The Cairo International Book Fair opened its doors, welcoming hundreds of publishers from around the world displaying literary and academic works. The cultural program includes poetry readings, panel discussions with acclaimed authors, and dedicated children's pavilions to foster reading habits.",
            metadata={"doc_id": "doc_en_culture_1", "language": "en", "expected_category": "Culture"}
        ),
        Document(
            page_content="An independent feature film won top honors at the International Film Festival following widespread critical praise. Jurors commended the director's narrative style, cinematography, and poignant portrayal of contemporary social issues, which resonated deeply with international audiences.",
            metadata={"doc_id": "doc_en_culture_2", "language": "en", "expected_category": "Culture"}
        ),
        Document(
            page_content="The Ministry of Culture organized a grand musical concert at the Opera House to honor classical heritage and legendary composers. The national orchestra delivered contemporary arrangements of traditional orchestral pieces, blending historical musical heritage with modern symphonic elements.",
            metadata={"doc_id": "doc_en_culture_3", "language": "en", "expected_category": "Culture"}
        ),
        Document(
            page_content="Archaeologists uncovered an ancient tomb dating back to the Eighteenth Dynasty near Luxor containing well-preserved polychrome wall murals. Excavations revealed burial artifacts, statues, and ritual vessels that offer valuable historical insights into ancient funerary customs.",
            metadata={"doc_id": "doc_en_culture_4", "language": "en", "expected_category": "Culture"}
        ),

        # ============================================================
        # --- FRENCH (20 CHUNKS - 4 PER CATEGORY) ---
        # ============================================================
        # Health (4)
        Document(
            page_content="Le ministère de la Santé a annoncé le lancement officiel d'une vaste campagne nationale de vaccination ciblant les enfants de moins de cinq ans dans toutes les provinces. L'initiative déploie des unités médicales mobiles dans les zones rurales isolées afin de fournir des vaccins gratuits, renforcer l'immunité collective et réduire considérablement les hospitalisations pédiatriques saisonnières.",
            metadata={"doc_id": "doc_fr_health_1", "language": "fr", "expected_category": "Health"}
        ),
        Document(
            page_content="Une étude médicale récente indique que la pratique quotidienne d'une activité physique modérée comme la marche rapide pendant trente minutes réduit le risque de maladies cardiovasculaires de près de quarante pour cent. Les chercheurs soulignent que l'adoption d'un mode de vie actif constitue une mesure préventive essentielle contre le diabète de type 2 et l'hypertension.",
            metadata={"doc_id": "doc_fr_health_2", "language": "fr", "expected_category": "Health"}
        ),
        Document(
            page_content="Les hôpitaux universitaires ont inauguré un service d'oncologie pédiatrique moderne équipé d'appareils de radiothérapie et d'imagerie par résonance magnétique de dernière génération. Cet établissement propose des protocoles de traitement gratuits aux patients défavorisés afin de réduire les listes d'attente et d'offrir un accompagnement psychologique aux familles.",
            metadata={"doc_id": "doc_fr_health_3", "language": "fr", "expected_category": "Health"}
        ),
        Document(
            page_content="L'Organisation mondiale de la santé alerte dans son dernier rapport sur la hausse inquiétante de l'obésité chez les adolescents en raison de la sédentarité et de la consommation d'aliments ultra-transformés. L'organisation appelle les gouvernements à imposer des restrictions strictes sur la publicité alimentaire ciblant les jeunes et à promouvoir la nutrition dans les écoles.",
            metadata={"doc_id": "doc_fr_health_4", "language": "fr", "expected_category": "Health"}
        ),

        # Sports (4)
        Document(
            page_content="L'équipe nationale égyptienne de football s'est imposée deux buts à un face à la Tunisie lors d'un match amical international disputé au stade international du Caire. Le but de la victoire a été inscrit dans les dernières minutes de la rencontre à la suite d'une contre-attaque rapide, renforçant le moral de l'équipe avant les éliminatoires continentales.",
            metadata={"doc_id": "doc_fr_sports_1", "language": "fr", "expected_category": "Sports"}
        ),
        Document(
            page_content="Un nageur égyptien a décroché la médaille d'or sur le 100 mètres papillon lors des Championnats du monde de natation organisés à Berlin. L'athlète a battu le record du championnat de quelques fractions de seconde, affirmant sa position de favori pour les prochains Jeux olympiques.",
            metadata={"doc_id": "doc_fr_sports_2", "language": "fr", "expected_category": "Sports"}
        ),
        Document(
            page_content="Le club d'Al Ahly a officialisé le recrutement d'un meneur de jeu brésilien pour une durée de quatre saisons. L'entraîneur principal s'est réjoui de cette signature, soulignant la polyvalence tactique et les qualités offensives du joueur pour renforcer l'effectif lors des compétitions nationales et africaines à venir.",
            metadata={"doc_id": "doc_fr_sports_3", "language": "fr", "expected_category": "Sports"}
        ),
        Document(
            page_content="L'équipe nationale de handball s'est qualifiée pour les demi-finales du championnat d'Afrique après avoir battu l'Algérie 30 à 25 au terme d'un match intense. Les arrêts décisifs du gardien dans les dernières minutes ont permis de maintenir l'avantage et d'assurer une place en finale.",
            metadata={"doc_id": "doc_fr_sports_4", "language": "fr", "expected_category": "Sports"}
        ),

        # Economy (4)
        Document(
            page_content="Les indices de la bourse ont clôturé en hausse aujourd'hui, portés par des achats massifs d'investisseurs institutionnels étrangers. Les secteurs de l'immobilier et des services financiers ont mené cette progression à la suite d'annonces de nouveaux partenariats d'investissement et d'une amélioration de la liquidité sur le marché.",
            metadata={"doc_id": "doc_fr_economy_1", "language": "fr", "expected_category": "Economy"}
        ),
        Document(
            page_content="Le comité de politique monétaire de la Banque centrale a décidé de maintenir ses taux d'intérêt directeurs inchangés afin de préserver la stabilité macroéconomique et de contenir l'inflation. Les responsables ont souligné que la politique monétaire restera prudente pour soutenir la croissance et stabiliser les prix à la consommation.",
            metadata={"doc_id": "doc_fr_economy_2", "language": "fr", "expected_category": "Economy"}
        ),
        Document(
            page_content="Le ministère des Finances a signé un accord d'emprunt d'un milliard de dollars avec la Banque mondiale pour moderniser les infrastructures nationales et le réseau d'énergie renouvelable. Ce projet soutient les partenariats public-privé visant à développer l'énergie solaire et éolienne tout en réduisant la dépendance aux combustibles fossiles.",
            metadata={"doc_id": "doc_fr_economy_3", "language": "fr", "expected_category": "Economy"}
        ),
        Document(
            page_content="Les exportations agricoles ont enregistré une hausse de quinze pour cent au cours du premier semestre par rapport à la même période de l'année précédente. L'ouverture de nouveaux marchés internationaux et l'application de normes de qualité strictes ont stimulé la demande mondiale pour les produits locaux.",
            metadata={"doc_id": "doc_fr_economy_4", "language": "fr", "expected_category": "Economy"}
        ),

        # Technology (4)
        Document(
            page_content="Une jeune entreprise technologique a lancé une plateforme d'intelligence artificielle basée sur des algorithmes d'apprentissage profond pour optimiser la consommation d'énergie des bâtiments intelligents. En analysant les données des capteurs en temps réel, le logiciel prévoit les pics de charge et ajuste automatiquement la climatisation.",
            metadata={"doc_id": "doc_fr_tech_1", "language": "fr", "expected_category": "Technology"}
        ),
        Document(
            page_content="Des ingénieurs de l'aérospatiale ont lancé avec succès un prototype de satellite miniature conçu pour évaluer les communications en orbite basse. La mission teste la connectivité des objets connectés et la fiabilité des transmissions de données face aux radiations spatiales.",
            metadata={"doc_id": "doc_fr_tech_2", "language": "fr", "expected_category": "Technology"}
        ),
        Document(
            page_content="L'autorité de régulation des télécommunications a commencé à attribuer les fréquences 5G aux opérateurs mobiles du pays. Le haut débit et la très faible latence de la 5G devraient accélérer le développement de la télémédecine, des transports autonomes et de l'automatisation industrielle dans les grandes villes.",
            metadata={"doc_id": "doc_fr_tech_3", "language": "fr", "expected_category": "Technology"}
        ),
        Document(
            page_content="Une entreprise technologique mondiale a dévoilé un processeur quantique de nouvelle génération conçu pour la simulation de données complexes et la modélisation moléculaire. Ce composant effectue des calculs bien plus rapidement que les supercalculateurs traditionnels, ouvrant la voie à des avancées en pharmacologie.",
            metadata={"doc_id": "doc_fr_tech_4", "language": "fr", "expected_category": "Technology"}
        ),

        # Culture (4)
        Document(
            page_content="Le Foire internationale du livre du Caire a ouvert ses portes en accueillant des centaines d'éditeurs venus du monde entier pour présenter leurs dernières publications littéraires et académiques. Le programme culturel comprend des lectures de poésie, des conférences d'auteurs et des ateliers pour enfants.",
            metadata={"doc_id": "doc_fr_culture_1", "language": "fr", "expected_category": "Culture"}
        ),
        Document(
            page_content="Un long métrage indépendant a remporté le grand prix du Festival international du film après avoir suscité l'enthousiasme de la critique. Le jury a salué la réalisation, la photographie et la peinture émouvante des enjeux sociaux contemporains abordés dans l'œuvre.",
            metadata={"doc_id": "doc_fr_culture_2", "language": "fr", "expected_category": "Culture"}
        ),
        Document(
            page_content="Le ministère de la Culture a organisé un grand concert symphonique à l'Opéra pour rendre hommage au patrimoine musical classique. L'orchestre national a interprété des arrangements modernes d'œuvres traditionnelles, alliant héritage historique et sensibilité musicale contemporaine.",
            metadata={"doc_id": "doc_fr_culture_3", "language": "fr", "expected_category": "Culture"}
        ),
        Document(
            page_content="Des archéologues ont découvert une tombe antique datant de la dix-huitième dynastie près de Louxor, renfermant des peintures murales très bien conservées. Les fouilles ont mis au jour des statues et des vases canopes qui apportent un éclairage précieux sur les rituels funéraires anciens.",
            metadata={"doc_id": "doc_fr_culture_4", "language": "fr", "expected_category": "Culture"}
        ),
    ]


# ============================================================
# PROMPTS
# ============================================================
SIMPLE_EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a multilingual news analyst.
Write a summary in 2-3 sentences MAXIMUM (no more than 60 words).
Return ONLY a valid JSON object with fields "summary" and "category".
Allowed categories: Politics, Economy, Sports, Technology, Culture, Health, Other.
Do not add markdown or explanations.""",
        ),
        ("human", "Requested language: {language_name}\nText:\n{text}"),
    ]
)

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an unbiased AI judge evaluating the quality of a news summary.

Evaluate the generated summary against the source text using these criteria.

1. "faithfulness"
Evaluate whether the summary accurately represents the source text without hallucinations, invented facts, contradictions, or incorrect information.

2. "coverage"
Evaluate whether the summary includes the main ideas, important facts, events, entities, and conclusions from the source text.

3. "quality"
Evaluate the overall summary quality, including clarity, coherence, relevance, conciseness, and readability.

4. "language_consistency"
Evaluate whether the summary is written in the same language as the source text.

For each criterion:
- Give a score from 1 to 5.
- Provide a short reason explaining the score.

Scoring guide:
5 = Excellent. The summary fully satisfies the criterion with no meaningful issues.
4 = Good. The summary satisfies the criterion with only minor issues.
3 = Acceptable. The summary is partially successful but has noticeable weaknesses.
2 = Poor. The summary has significant problems.
1 = Very poor. The summary fails the criterion.

Return ONLY a valid JSON object with exactly these fields:

{{
    "faithfulness": {{
        "score": 1-5,
        "reason": "short explanation"
    }},
    "coverage": {{
        "score": 1-5,
        "reason": "short explanation"
    }},
    "quality": {{
        "score": 1-5,
        "reason": "short explanation"
    }},
    "language_consistency": {{
        "score": 1-5,
        "reason": "short explanation"
    }},
    "overall_reason": "short explanation of the overall evaluation"
}}

Do not include markdown.
Do not include any text outside the JSON object."""
        ),
        (
            "human",
            "Source Text:\n{source_text}\n\nGenerated Summary:\n{summary}",
        ),
    ]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def extract_json(raw_content: str):
    """Robust JSON extractor handling markdown, single quotes, trailing commas, and unquoted keys."""
    if not raw_content:
        raise ValueError("Empty model response.")

    cleaned = raw_content.strip()

    # 1. Remove markdown code fences if present
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    # 2. Try standard parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Extract JSON object structure via regex
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not find JSON object in response: {raw_content[:200]}"
        )

    json_str = match.group(0)

    # 4. Clean trailing commas inside JSON objects/arrays
    json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 5. Fix single quotes to double quotes for keys and values
    fixed = re.sub(
        r"([{\s,])'([^'\\]*(?:\\.[^'\\]*)*)'(\s*:)", r'\1"\2"\3', json_str
    )
    fixed = re.sub(
        r"(:\s*)'([^'\\]*(?:\\.[^'\\]*)*)'([\s,\}])", r'\1"\2"\3', fixed
    )

    # 6. Fix unquoted keys (e.g. {faithfulness: 5})
    fixed = re.sub(r"([{\s,])([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', fixed)

    # Final parse attempt
    return json.loads(fixed)


def verify_language(summary_text: str, expected_lang_code: str) -> bool:
    """Language Accuracy Measurement using langdetect."""
    try:
        detected = detect(summary_text)
        return detected == expected_lang_code
    except Exception:
        return False


def evaluate_summary_quality_with_retry(
    source_text: str, summary_text: str, max_retries=5
) -> dict:
    """Summary Quality Measurement using LLM-as-a-Judge with Retry Mechanism for Rate Limits."""
    delay = 10

    for attempt in range(max_retries):
        try:
            if not JUDGE_MODEL or not summary_text:
                return {
                    "status": "FAILED",
                    "quality": None,
                    "faithfulness": None,
                    "coverage": None,
                    "language_consistency": None,
                }

            bound_judge = JUDGE_MODEL.bind(
                response_format={"type": "json_object"}
            )
            chain = JUDGE_PROMPT | bound_judge

            response = chain.invoke(
                {"source_text": source_text, "summary": summary_text}
            )

            data = extract_json(str(response.content))

            def get_score_and_reason(field_name):
                val = data.get(field_name, {})
                if isinstance(val, dict):
                    score = int(val.get("score", 0))
                    reason = str(val.get("reason", ""))
                else:
                    score = int(val) if str(val).isdigit() else 0
                    reason = ""
                return score, reason

            faithfulness, f_reason = get_score_and_reason("faithfulness")
            coverage, c_reason = get_score_and_reason("coverage")
            quality, q_reason = get_score_and_reason("quality")
            lang_cons, l_reason = get_score_and_reason("language_consistency")

            return {
                "status": "SUCCESS",
                "faithfulness": faithfulness,
                "coverage": coverage,
                "quality": quality,
                "language_consistency": lang_cons,
                "faithfulness_reason": f_reason,
                "coverage_reason": c_reason,
                "quality_reason": q_reason,
                "language_consistency_reason": l_reason,
                "overall_reason": str(data.get("overall_reason", "")),
            }

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg:
                # Try to extract the wait time Groq suggests (e.g. "7m12.864s")
                match_min = re.search(r"(\d+)m([\d\.]+)s", err_msg)
                if match_min:
                    minutes = int(match_min.group(1))
                    seconds = float(match_min.group(2))
                    wait_time = (minutes * 60) + seconds + 5  # +5s safety margin
                else:
                    wait_time = delay

                print(
                    f"\n[JUDGE RATE LIMIT] Attempt {attempt+1}/{max_retries}. Waiting {round(wait_time)}s for Judge API..."
                )
                time.sleep(wait_time)
                delay *= 2
                continue

            print(f"[JUDGE BYPASS]: Skipped chunk evaluation due to issue: {e}")
            break

    return {
        "status": "FAILED",
        "faithfulness": None,
        "coverage": None,
        "quality": None,
        "language_consistency": None,
        "faithfulness_reason": "",
        "coverage_reason": "",
        "quality_reason": "",
        "language_consistency_reason": "",
        "overall_reason": "",
    }


def run_evaluation(model, chunk_text: str, language_name: str):
    """Candidate LLM execution with latency tracking and forced JSON mode."""
    try:
        bound_model = model.bind(response_format={"type": "json_object"})
        chain = SIMPLE_EVAL_PROMPT | bound_model

        start_time = time.perf_counter()
        response = chain.invoke(
            {"text": chunk_text, "language_name": language_name}
        )
        elapsed = time.perf_counter() - start_time

        data = extract_json(str(response.content))
        summary = str(data.get("summary", "")).strip()
        category = str(data.get("category", "")).strip()

        if not summary or not category:
            raise ValueError("Model returned an empty summary or category.")

        return {
            "summary": summary,
            "category": category,
            "time": elapsed,
            "error": False,
            "error_message": None,
        }

    except Exception as e:
        return {
            "summary": "",
            "category": "",
            "time": None,
            "error": True,
            "error_message": str(e),
        }


def run_evaluation_with_retry(model, chunk_text, language_name, max_retries=4):
    delay = 5  # seconds; doubles each retry
    result = None
    for attempt in range(max_retries):
        result = run_evaluation(model, chunk_text, language_name)
        if not result["error"]:
            return result
        if "429" in str(result["error_message"]) or "rate_limit" in str(
            result["error_message"]
        ):
            print(
                f"[RATE LIMIT] attempt {attempt+1}/{max_retries}, waiting {delay}s..."
            )
            time.sleep(delay)
            delay *= 2
            continue
        return result  # non-rate-limit error: fail immediately, don't retry
    return result  # exhausted retries


# ============================================================
# PRINT RESULTS & METRICS
# ============================================================


def print_evaluation_metrics(metrics_by_lang):
    best_models = {}

    print("\n" + "=" * 120)
    print(" DETAILED BENCHMARK METRICS BY LANGUAGE")
    print("=" * 120)

    for lang in ["AR", "EN", "FR"]:
        print(f"\nLanguage: {lang}\n")
        print(
            f"{'Model':<16} {'Lang Acc':<10} {'Class Acc':<11} {'Quality (1-5)':<15} "
            f"{'Faithful (1-5)':<15} {'Eff.Quality':<12} {'Error Rate':<12} {'Avg Time':<10} {'Success':<8}"
        )
        print("-" * 115)

        best_model_name = None
        # tie-break order: effective_quality, then class_acc, then lang_acc, then -avg_time
        best_score = (-1.0, -1.0, -1.0, float("-inf"))

        for model_name, stats in metrics_by_lang[lang].items():
            total = stats["total"]
            if total == 0:
                continue

            success = stats["success_runs"]
            error_count = stats["errors"]
            eval_success_count = stats["evaluated_quality_runs"]

            lang_acc = (stats["lang_ok"] / total) * 100
            class_acc = (stats["category_ok"] / total) * 100
            error_rate = (error_count / total) * 100
            avg_time = stats["total_time"] / success if success > 0 else 0.0

            avg_quality = (
                stats["quality_score_sum"] / eval_success_count
                if eval_success_count > 0
                else 0.0
            )
            avg_faithfulness = (
                stats["faithfulness_score_sum"] / eval_success_count
                if eval_success_count > 0
                else 0.0
            )

            # Effective quality penalizes hallucination: a model that writes
            # fluently (high "quality") but invents facts (low "faithfulness")
            # can no longer top the ranking on style/readability alone.
            # Divided by 5 to keep the result on the same 0-5 scale as the
            # other columns, instead of a raw 0-25 product.
            effective_quality = (
                (avg_quality * avg_faithfulness) / 5
                if eval_success_count > 0
                else 0.0
            )

            print(
                f"{model_name:<16} {lang_acc:>6.0f}%    {class_acc:>7.0f}%     "
                f"{avg_quality:>8.2f} / 5     {avg_faithfulness:>8.2f} / 5     "
                f"{effective_quality:>8.2f}     {error_rate:>8.0f}%     "
                f"{avg_time:>6.2f}s    {success}/{total}"
            )

            current_score = (effective_quality, class_acc, lang_acc, -avg_time)
            if current_score > best_score:
                best_score = current_score
                best_model_name = model_name

        best_models[lang] = best_model_name
        print("-" * 115)

    print("\n" + "=" * 120)
    print(" BEST MODEL SELECTION PER LANGUAGE (ranked by Effective Quality = Quality x Faithfulness / 5)")
    print("=" * 120)
    print(f" Best model for Arabic   (AR) : {best_models.get('AR', 'N/A')}")
    print(f" Best model for English  (EN) : {best_models.get('EN', 'N/A')}")
    print(f" Best model for French   (FR) : {best_models.get('FR', 'N/A')}")
    print("=" * 120 + "\n")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 120)
    print("MULTILINGUAL LLM BENCHMARKING")
    print("=" * 120)

    dummy_chunks = get_extended_dummy_chunks()

    if not CANDIDATE_MODELS:
        print(
            "\nNo candidate models available. Please set API Keys in your environment."
        )
        raise SystemExit(1)

    metrics_by_lang = defaultdict(
        lambda: defaultdict(
            lambda: {
                "total": 0,
                "success_runs": 0,
                "errors": 0,
                "lang_ok": 0,
                "category_ok": 0,
                "quality_score_sum": 0.0,
                "faithfulness_score_sum": 0.0,  # NEW: tracks faithfulness alongside quality
                "evaluated_quality_runs": 0,
                "total_time": 0.0,
            }
        )
    )

    detailed_file_records = []

    for index, chunk in enumerate(dummy_chunks, start=1):
        metadata = chunk.metadata
        doc_id = metadata["doc_id"]
        true_language = metadata["language"]
        true_category = metadata["expected_category"]
        language_name = LANGUAGE_NAMES[true_language]
        lang_key = true_language.upper()

        print(
            f"--> Processing Article {index}/{len(dummy_chunks)} [{lang_key} - Category: {true_category}]..."
        )

        for model_name, model in CANDIDATE_MODELS.items():
            result = run_evaluation_with_retry(
                model=model,
                chunk_text=chunk.page_content,
                language_name=language_name,
            )

            stats = metrics_by_lang[lang_key][model_name]
            stats["total"] += 1

            if result["error"]:
                stats["errors"] += 1
                record = {
                    "doc_id": doc_id,
                    "language": true_language,
                    "model": model_name,
                    "status": "ERROR",
                    "error_message": result["error_message"],
                    "summary": None,
                }
            else:
                stats["success_runs"] += 1
                stats["total_time"] += result["time"]

                # 1. Verify Language Accuracy
                lang_ok = verify_language(result["summary"], true_language)
                if lang_ok:
                    stats["lang_ok"] += 1

                # 2. Verify Category Classification Accuracy
                category_ok = (
                    result["category"].strip().lower()
                    == true_category.strip().lower()
                )
                if category_ok:
                    stats["category_ok"] += 1

                # 3. Evaluate Summary Quality via LLM Judge (with retry)
                quality_scores = evaluate_summary_quality_with_retry(
                    chunk.page_content, result["summary"]
                )

                # Only accumulate when the judge call actually succeeded,
                # so a failed judge call doesn't silently pull averages down.
                if (
                    quality_scores.get("status") == "SUCCESS"
                    and quality_scores.get("quality") is not None
                ):
                    stats["quality_score_sum"] += quality_scores["quality"]
                    stats["evaluated_quality_runs"] += 1

                    # NEW: accumulate faithfulness in lockstep with quality
                    # so effective_quality can be computed later.
                    if quality_scores.get("faithfulness") is not None:
                        stats["faithfulness_score_sum"] += quality_scores[
                            "faithfulness"
                        ]

                record = {
                    "doc_id": doc_id,
                    "language": true_language,
                    "expected_category": true_category,
                    "model": model_name,
                    "status": "SUCCESS",
                    "predicted_category": result["category"],
                    "category_match": category_ok,
                    "language_match": lang_ok,
                    "judge_scores": quality_scores,
                    "latency_seconds": round(result["time"], 2),
                    "summary": result["summary"],
                }

            detailed_file_records.append(record)

            # Essential sleep duration to respect API Rate Limits and prevent Socket drops
            time.sleep(2.5)

    # Save output to JSON
    output_filename = "evaluation_results_last.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_file_records, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Summaries & judge scores saved to '{output_filename}'")

    # Display execution dashboard
    print_evaluation_metrics(metrics_by_lang)

    print("=" * 120)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 120)