"""
Enhanced STEM Tutor Bot for High School Students
Features: Multiple datasets, expanded knowledge base, semantic search, step-by-step solutions

Requirements:
pip install transformers torch datasets sentence-transformers numpy
"""
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import warnings
import re
import numpy as np
warnings.filterwarnings('ignore')

class EnhancedSTEMTutorBot:
    SIMILARITY_THRESHOLD = 0.45
    HIGH_CONFIDENCE_THRESHOLD = 0.65
    
    def __init__(self):
        """Initialize models and datasets"""
        print("=" * 70)
        print("  🎓 ENHANCED STEM TUTOR BOT - High School Edition")
        print("=" * 70)
        print("\n🔧 Initializing AI models and datasets...\n")
        
        # Load semantic similarity model
        print("📊 Loading semantic search model...")
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic model loaded!")
        except Exception as e:
            print(f"⚠️ Semantic model failed: {e}")
            self.semantic_model = None
        
        # Load QA model
        print("\n🤖 Loading BERT QA model...")
        model_name = "distilbert-base-cased-distilled-squad"
        self.qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ BERT model loaded!")
        
        # Load datasets
        self.datasets = {}
        self._load_datasets()
        
        # Conversation context
        self.conversation_history = []
        self.max_history = 5
        self.last_subject = None
        self.last_topic = None
        
        # Initialize expanded knowledge base
        self.knowledge_base = self._build_comprehensive_kb()
        
        # Pre-encode knowledge base for semantic search
        if self.semantic_model:
            print("\n🔍 Pre-encoding knowledge base for fast semantic search...")
            self._precompute_embeddings()
            print("✅ Knowledge base indexed!")
        
        # Subject detection keywords
        self.subject_keywords = {
            'physics': ['force', 'motion', 'energy', 'newton', 'gravity', 'mass', 'velocity', 
                       'acceleration', 'momentum', 'friction', 'wave', 'light', 'electricity',
                       'magnetism', 'pressure', 'work', 'power', 'thermodynamics'],
            'chemistry': ['atom', 'molecule', 'chemical', 'reaction', 'bond', 'element', 
                         'compound', 'acid', 'base', 'periodic', 'ion', 'electron', 'proton',
                         'neutron', 'covalent', 'ionic', 'oxidation', 'reduction', 'mole',
                         'stoichiometry', 'ph', 'catalyst'],
            'biology': ['cell', 'dna', 'gene', 'evolution', 'organism', 'photosynthesis', 
                       'respiration', 'protein', 'mitosis', 'meiosis', 'enzyme', 'ecosystem',
                       'species', 'bacteria', 'virus', 'tissue', 'organ', 'genetics'],
            'math': ['equation', 'algebra', 'geometry', 'calculus', 'derivative', 'integral',
                    'function', 'graph', 'polynomial', 'trigonometry', 'sine', 'cosine',
                    'pythagorean', 'quadratic', 'linear', 'slope', 'angle', 'triangle']
        }
        
        print("\n✨ Bot ready! All systems operational.\n")
    
    def _load_datasets(self):
        """Load multiple STEM datasets"""
        print("\n📚 Loading datasets...\n")
        
        # ScienceQA
        try:
            print("📖 Loading ScienceQA (21K+ multimodal questions)...")
            self.datasets['scienceqa'] = load_dataset("derek-thomas/ScienceQA", split="train")
            print(f"✅ ScienceQA: {len(self.datasets['scienceqa'])} questions loaded")
        except Exception as e:
            print(f"⚠️ ScienceQA failed: {e}")
            self.datasets['scienceqa'] = None
        
        # SciQ
        try:
            print("\n📖 Loading SciQ (13K+ science Q&A)...")
            self.datasets['sciq'] = load_dataset("allenai/sciq", split="train")
            print(f"✅ SciQ: {len(self.datasets['sciq'])} questions loaded")
        except Exception as e:
            print(f"⚠️ SciQ failed: {e}")
            self.datasets['sciq'] = None
        
        # MMLU (high school subjects)
        try:
            print("\n📖 Loading MMLU high school subjects...")
            mmlu_subjects = ['high_school_physics', 'high_school_chemistry', 
                           'high_school_biology', 'high_school_mathematics']
            self.datasets['mmlu'] = {}
            for subject in mmlu_subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                    self.datasets['mmlu'][subject] = ds
                    print(f"  ✅ {subject}: {len(ds)} questions")
                except:
                    print(f"  ⚠️ {subject}: failed to load")
            if self.datasets['mmlu']:
                print(f"✅ MMLU loaded successfully")
        except Exception as e:
            print(f"⚠️ MMLU failed: {e}")
            self.datasets['mmlu'] = None
    
    def _build_comprehensive_kb(self):
        """Build comprehensive high school STEM knowledge base"""
        kb = {
            'physics': {
                'motion': {
                    'keywords': ['motion', 'movement', 'moving', 'what is motion'],
                    'content': """**Motion** is the change in position of an object over time relative to a reference point.

**Key Concepts:**
• **Displacement**: Change in position (vector - has direction)
• **Velocity**: Rate of change of position with direction (v = Δx/Δt)
• **Speed**: How fast an object moves (scalar - no direction)
• **Acceleration**: Rate of change of velocity (a = Δv/Δt)

**Types of Motion:**
1. Linear/Translational: Movement in a straight line
2. Rotational: Spinning around an axis
3. Oscillatory: Back and forth (like a pendulum)
4. Circular: Moving in a circular path

**Important Note:** Motion is relative - you must measure it compared to a reference frame. You're stationary relative to your chair, but moving at 1,670 km/h relative to Earth's center as it rotates!"""
                },
                'kinetic_energy': {
                    'keywords': ['kinetic energy', 'ke', 'energy of motion'],
                    'content': """**Kinetic Energy (KE)** is the energy an object possesses due to its motion.

**Formula:** KE = ½mv²
• m = mass (kg)
• v = velocity (m/s)

**Key Points:**
• KE increases with the SQUARE of velocity - double the speed = 4× the energy
• All moving objects have kinetic energy
• Measured in Joules (J)

**Example Problem:**
A 1,000 kg car travels at 20 m/s. What's its kinetic energy?
KE = ½(1,000)(20²) = ½(1,000)(400) = 200,000 J = 200 kJ

**Real-world Connection:**
This quadratic relationship explains why high-speed crashes are so dangerous - a car at 100 km/h has 4× the kinetic energy of the same car at 50 km/h!"""
                },
                'newtons_first_law': {
                    'keywords': ['first law', 'newton first', 'law of inertia', 'inertia'],
                    'content': """**Newton's First Law (Law of Inertia):**
An object at rest stays at rest, and an object in motion stays in motion with constant velocity, unless acted upon by an unbalanced force.

**What is Inertia?**
Inertia is the tendency of objects to resist changes in motion. Mass is a measure of inertia - more massive objects have more inertia.

**Key Examples:**
• A book on a table won't move unless pushed (at rest stays at rest)
• A hockey puck on ice keeps sliding (in motion stays in motion)
• When a car brakes suddenly, passengers lurch forward (bodies want to keep moving)

**Common Misconception:**
Objects don't naturally slow down - they slow down because of friction and air resistance (external forces). In space with no friction, an object would keep moving forever!"""
                },
                'newtons_second_law': {
                    'keywords': ['second law', 'newton second', 'f=ma', 'force equals'],
                    'content': """**Newton's Second Law:**
F = ma (Force equals mass times acceleration)

**Breaking it down:**
• F = net force (Newtons, N)
• m = mass (kilograms, kg)
• a = acceleration (m/s²)

**Key Insights:**
1. Force and acceleration are directly proportional
2. Mass and acceleration are inversely proportional
3. Direction matters - force and acceleration are vectors

**Example:**
Push a 10 kg box with 50 N of force. What's the acceleration?
a = F/m = 50/10 = 5 m/s²

**Real Application:**
Why are sports cars fast? Either high force (powerful engine) or low mass (lightweight materials) gives high acceleration!"""
                },
                'newtons_third_law': {
                    'keywords': ['third law', 'newton third', '3rd law', 'action reaction', 'equal and opposite'],
                    'content': """**Newton's Third Law of Motion:**
For every action, there is an equal and opposite reaction.

**More Precisely:**
When object A exerts a force on object B, object B simultaneously exerts a force equal in magnitude and opposite in direction on object A.

**Key Points:**
• Forces always come in pairs (action-reaction pairs)
• The forces act on DIFFERENT objects
• Forces are equal in magnitude but opposite in direction
• Forces occur at the same time

**Examples:**
1. **Rocket propulsion**: Rocket pushes gas downward (action) → Gas pushes rocket upward (reaction)
2. **Walking**: You push Earth backward (action) → Earth pushes you forward (reaction)
3. **Swimming**: You push water backward (action) → Water pushes you forward (reaction)
4. **Book on table**: Book pushes down on table (action) → Table pushes up on book (reaction)

**Common Misconception:**
Action-reaction forces do NOT cancel out because they act on different objects! The book doesn't float because both forces act on the book - only the reaction force (table pushing up) acts on the book.

**Formula:** F₁₂ = -F₂₁
(Force on object 1 by object 2 equals negative force on object 2 by object 1)"""
                },
                'gravity': {
                    'keywords': ['gravity', 'gravitational', 'weight', 'g'],
                    'content': """**Gravity** is the attractive force between objects with mass.

**On Earth's Surface:**
• g = 9.8 m/s² (acceleration due to gravity)
• Weight = mg (weight is a force!)

**Important Distinction:**
• **Mass**: Amount of matter (kg) - doesn't change
• **Weight**: Gravitational force (N) - changes with location

**Example:**
A 60 kg person on Earth:
Weight = mg = 60 × 9.8 = 588 N

Same person on the Moon (g = 1.6 m/s²):
Weight = 60 × 1.6 = 96 N
Mass still = 60 kg!

**Newton's Law of Universal Gravitation:**
F = G(m₁m₂)/r²
Every object attracts every other object - but the force is only noticeable for very massive objects."""
                }
            },
            'chemistry': {
                'atom': {
                    'keywords': ['atom', 'atomic', 'what is atom'],
                    'content': """**Atom** is the smallest unit of matter that retains the properties of an element.

**Structure:**
• **Nucleus** (center): Contains protons (+) and neutrons (neutral)
• **Electron Cloud**: Electrons (-) orbit the nucleus in energy levels

**Subatomic Particles:**
1. **Protons**: Positive charge, mass ≈ 1 amu, defines the element
2. **Neutrons**: No charge, mass ≈ 1 amu, affects atomic mass
3. **Electrons**: Negative charge, mass ≈ 0 amu, involved in bonding

**Key Numbers:**
• **Atomic Number (Z)**: Number of protons
• **Mass Number (A)**: Protons + neutrons
• **Isotopes**: Same element, different number of neutrons

**Example:** Carbon-12 (¹²C)
• 6 protons (atomic number)
• 6 neutrons (12 - 6)
• 6 electrons (neutral atom)"""
                },
                'covalent_bond': {
                    'keywords': ['covalent', 'covalent bond', 'sharing electrons'],
                    'content': """**Covalent Bond** forms when atoms share electrons to achieve stable electron configurations.

**How It Works:**
Atoms share pairs of electrons to fill their outer electron shells (usually want 8 electrons - octet rule).

**Types:**
• **Single Bond**: Share 1 pair (2 electrons) - Example: H-H
• **Double Bond**: Share 2 pairs (4 electrons) - Example: O=O
• **Triple Bond**: Share 3 pairs (6 electrons) - Example: N≡N

**Polarity:**
• **Nonpolar**: Equal sharing (same electronegativity) - Example: H₂, O₂
• **Polar**: Unequal sharing (different electronegativity) - Example: H₂O

**Example - Water (H₂O):**
Oxygen shares electrons with 2 hydrogen atoms. Oxygen is more electronegative, so electrons spend more time near O, making it δ- and H atoms δ+. This creates a polar molecule!"""
                },
                'ionic_bond': {
                    'keywords': ['ionic', 'ionic bond', 'transfer electrons'],
                    'content': """**Ionic Bond** forms when electrons are transferred from one atom to another, creating oppositely charged ions that attract.

**Process:**
1. Metal loses electrons → becomes positive cation
2. Nonmetal gains electrons → becomes negative anion
3. Opposite charges attract (electrostatic force)

**Example - Sodium Chloride (NaCl):**
• Na (sodium): 11 electrons → loses 1 → Na⁺ (10 electrons)
• Cl (chlorine): 17 electrons → gains 1 → Cl⁻ (18 electrons)
• Na⁺ and Cl⁻ attract to form NaCl (table salt)

**Properties of Ionic Compounds:**
• High melting/boiling points
• Conduct electricity when dissolved in water (ions are mobile)
• Form crystalline structures
• Generally hard but brittle

**Remember:** Metals + Nonmetals = Ionic bonds"""
                },
                'ph_scale': {
                    'keywords': ['ph', 'ph scale', 'acidity', 'acidic', 'basic'],
                    'content': """**pH Scale** measures the acidity or basicity of a solution (0-14).

**Scale:**
• pH < 7: **Acidic** (more H⁺ ions)
• pH = 7: **Neutral** (pure water)
• pH > 7: **Basic/Alkaline** (more OH⁻ ions)

**Formula:** pH = -log[H⁺]
Each pH unit = 10× difference in H⁺ concentration

**Examples:**
• pH 1-2: Stomach acid, battery acid
• pH 3: Lemon juice, vinegar
• pH 7: Pure water, blood
• pH 8-9: Baking soda solution
• pH 13-14: Drain cleaner, lye

**Important:**
pH 4 is 10× more acidic than pH 5
pH 3 is 100× more acidic than pH 5

**Indicators:**
• Litmus paper: Red in acid, blue in base
• Phenolphthalein: Colorless in acid, pink in base"""
                }
            },
            'biology': {
                'photosynthesis': {
                    'keywords': ['photosynthesis', 'photosynthesize'],
                    'content': """**Photosynthesis** is the process plants use to convert light energy into chemical energy (glucose).

**Overall Equation:**
6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
(carbon dioxide + water + light → glucose + oxygen)

**Where:** Occurs in chloroplasts (contain chlorophyll)

**Two Main Stages:**

**1. Light-Dependent Reactions (in thylakoids):**
• Capture light energy
• Split water (H₂O) → releases O₂
• Produce ATP and NADPH

**2. Light-Independent Reactions/Calvin Cycle (in stroma):**
• Use ATP and NADPH
• Fix CO₂ into glucose
• Can occur without direct light

**Importance:**
• Produces oxygen we breathe
• Base of most food chains
• Stores solar energy in chemical bonds

**Factors Affecting Rate:**
Light intensity, CO₂ concentration, temperature, water availability"""
                },
                'cellular_respiration': {
                    'keywords': ['cellular respiration', 'respiration cell'],
                    'content': """**Cellular Respiration** is the process cells use to break down glucose and produce ATP (energy).

**Overall Equation:**
C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + ATP
(glucose + oxygen → carbon dioxide + water + energy)

**Note:** This is the OPPOSITE of photosynthesis!

**Three Stages:**

**1. Glycolysis (cytoplasm):**
• Breaks glucose into 2 pyruvate
• Produces 2 ATP (net) + 2 NADH
• Doesn't require oxygen

**2. Krebs Cycle/Citric Acid Cycle (mitochondrial matrix):**
• Processes pyruvate
• Produces CO₂, NADH, FADH₂
• Small amount of ATP

**3. Electron Transport Chain (inner mitochondrial membrane):**
• Uses NADH and FADH₂
• Produces MOST ATP (~34 ATP)
• Requires oxygen (aerobic)

**Total Yield:** ~38 ATP per glucose molecule

**Without Oxygen:** Cells do fermentation (only glycolysis) → much less efficient!"""
                },
                'dna': {
                    'keywords': ['dna', 'deoxyribonucleic'],
                    'content': """**DNA (Deoxyribonucleic Acid)** is the molecule that carries genetic information in all living organisms.

**Structure - Double Helix:**
• Two strands twisted together (discovered by Watson & Crick, 1953)
• Sugar-phosphate backbone (outside)
• Nitrogenous bases (inside, paired)

**Four Bases:**
• **Purines:** Adenine (A), Guanine (G) - larger, 2 rings
• **Pyrimidines:** Thymine (T), Cytosine (C) - smaller, 1 ring

**Base Pairing Rules (Chargaff's Rules):**
• A always pairs with T (2 hydrogen bonds)
• G always pairs with C (3 hydrogen bonds)

**Function:**
1. Stores genetic information
2. Passed from parents to offspring
3. Provides instructions for making proteins

**DNA vs RNA:**
• DNA: double-stranded, has thymine, deoxyribose sugar
• RNA: single-stranded, has uracil (not thymine), ribose sugar

**Organization:**
DNA → Genes → Chromosomes → Nucleus"""
                },
                'mitosis': {
                    'keywords': ['mitosis', 'cell division', 'mitotic'],
                    'content': """**Mitosis** is cell division that produces two identical daughter cells (for growth and repair).

**Purpose:**
• Growth and development
• Replace damaged cells
• Asexual reproduction (some organisms)

**Phases (PMAT):**

**1. Prophase:**
• Chromatin condenses into chromosomes
• Nuclear envelope breaks down
• Spindle fibers form

**2. Metaphase:**
• Chromosomes line up at cell's equator (metaphase plate)
• Spindle fibers attach to centromeres

**3. Anaphase:**
• Sister chromatids separate
• Move to opposite poles of cell

**4. Telophase:**
• Nuclear envelopes reform
• Chromosomes decondense
• Cytokinesis begins (cell splits)

**Result:** 2 diploid daughter cells, genetically identical to parent

**Mitosis vs Meiosis:**
• Mitosis: 1 division → 2 identical cells (somatic cells)
• Meiosis: 2 divisions → 4 different cells (gametes/sex cells)"""
                }
            },
            'math': {
                'quadratic_equation': {
                    'keywords': ['quadratic', 'quadratic equation', 'ax^2'],
                    'content': """**Quadratic Equation** is a polynomial equation of degree 2.

**Standard Form:** ax² + bx + c = 0
• a, b, c are constants (a ≠ 0)
• x is the variable

**Quadratic Formula:**
x = [-b ± √(b² - 4ac)] / 2a

**The Discriminant (b² - 4ac):**
• If > 0: Two real solutions
• If = 0: One real solution (repeated root)
• If < 0: No real solutions (two complex solutions)

**Example:** Solve x² - 5x + 6 = 0
a=1, b=-5, c=6
x = [5 ± √(25-24)] / 2 = [5 ± 1] / 2
x = 3 or x = 2

**Other Methods:**
• Factoring: (x-3)(x-2) = 0
• Completing the square
• Graphing (x-intercepts)

**Graph:** Parabola (U-shaped curve)
• Opens up if a > 0
• Opens down if a < 0"""
                },
                'pythagorean_theorem': {
                    'keywords': ['pythagorean', 'pythagoras', 'a^2 + b^2'],
                    'content': """**Pythagorean Theorem** relates the sides of a right triangle.

**Formula:** a² + b² = c²
• a, b = legs (sides forming the right angle)
• c = hypotenuse (longest side, opposite right angle)

**Example:**
Triangle with legs 3 and 4. Find hypotenuse.
3² + 4² = c²
9 + 16 = c²
25 = c²
c = 5

**Common Pythagorean Triples:**
• 3-4-5
• 5-12-13
• 8-15-17
• 7-24-25

**Converse:**
If a² + b² = c², then triangle IS a right triangle.

**Applications:**
• Finding distances
• Navigation
• Construction
• Computer graphics"""
                }
            }
        }
        return kb
    
    def _precompute_embeddings(self):
        """Pre-encode knowledge base entries for fast semantic search"""
        if not self.semantic_model:
            return
        
        self.kb_embeddings = {}
        for subject, topics in self.knowledge_base.items():
            self.kb_embeddings[subject] = {}
            for topic_key, topic_data in topics.items():
                text = f"{topic_key} {' '.join(topic_data['keywords'])} {topic_data['content']}"
                embedding = self.semantic_model.encode(text, convert_to_tensor=True)
                self.kb_embeddings[subject][topic_key] = embedding
    
    def detect_law_number(self, question):
        """Detect which numbered law is being asked about"""
        question_lower = question.lower()
        
        # Check for explicit numbers
        number_words = {
            'first': 1, '1st': 1, 'one': 1,
            'second': 2, '2nd': 2, 'two': 2,
            'third': 3, '3rd': 3, 'three': 3
        }
        
        for word, num in number_words.items():
            if word in question_lower:
                return num
        
        # Check for digits
        import re
        digits = re.findall(r'\b([1-3])\b', question_lower)
        if digits:
            return int(digits[0])
        
        return None
    
    def semantic_search_kb(self, question):
        """Use semantic similarity to find best KB match"""
        if not self.semantic_model:
            return None, None, 0
        
        # Special handling for numbered laws
        law_num = self.detect_law_number(question)
        if law_num and 'newton' in question.lower():
            law_map = {
                1: 'newtons_first_law',
                2: 'newtons_second_law',
                3: 'newtons_third_law'
            }
            if law_map.get(law_num):
                print(f"🎯 Detected Newton's Law #{law_num} - direct match")
                return 'physics', law_map[law_num], 0.95
        
        question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
        
        best_match = None
        best_subject = None
        best_score = 0
        
        for subject, topics in self.kb_embeddings.items():
            for topic_key, topic_embedding in topics.items():
                similarity = util.cos_sim(question_embedding, topic_embedding).item()
                if similarity > best_score:
                    best_score = similarity
                    best_match = topic_key
                    best_subject = subject
        
        if best_score > self.SIMILARITY_THRESHOLD:
            return best_subject, best_match, best_score
        
        return None, None, 0
    
    def search_scienceqa(self, question):
        """Enhanced ScienceQA search with semantic similarity"""
        if not self.datasets.get('scienceqa'):
            return None
        
        try:
            print("🔍 Searching ScienceQA dataset...")
            
            best_match = None
            best_score = 0
            
            # Use semantic search if available
            if self.semantic_model:
                question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
                
                # Sample subset for speed
                for i, example in enumerate(self.datasets['scienceqa']):
                    if i > 3000:
                        break
                    
                    example_text = example['question']
                    example_embedding = self.semantic_model.encode(example_text, convert_to_tensor=True)
                    similarity = util.cos_sim(question_embedding, example_embedding).item()
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = example
            
            if best_match and best_score > self.SIMILARITY_THRESHOLD:
                print(f"✅ Found match (similarity: {best_score:.2%})")
                
                choices = best_match.get('choices', [])
                answer_idx = best_match.get('answer', 0)
                correct_answer = choices[answer_idx] if answer_idx < len(choices) else ""
                
                lecture = best_match.get('lecture', '')
                solution = best_match.get('solution', '')
                
                answer_parts = []
                if correct_answer:
                    answer_parts.append(f"**Answer:** {correct_answer}")
                if lecture:
                    answer_parts.append(f"\n**📚 Explanation:**\n{lecture}")
                if solution:
                    answer_parts.append(f"\n**💡 Solution:**\n{solution}")
                
                return {
                    'answer': "\n".join(answer_parts) if answer_parts else correct_answer,
                    'source': 'ScienceQA',
                    'confidence': best_score,
                    'matched_question': best_match['question']
                }
            
            return None
            
        except Exception as e:
            print(f"ScienceQA error: {e}")
            return None
    
    def search_mmlu(self, question, subject_hint=None):
        """Search MMLU dataset"""
        if not self.datasets.get('mmlu'):
            return None
        
        try:
            print("🔍 Searching MMLU dataset...")
            
            # Determine which MMLU subject to search
            subjects_to_search = []
            if subject_hint == 'physics':
                subjects_to_search = ['high_school_physics']
            elif subject_hint == 'chemistry':
                subjects_to_search = ['high_school_chemistry']
            elif subject_hint == 'biology':
                subjects_to_search = ['high_school_biology']
            elif subject_hint == 'math':
                subjects_to_search = ['high_school_mathematics']
            else:
                subjects_to_search = list(self.datasets['mmlu'].keys())
            
            best_match = None
            best_score = 0
            
            if self.semantic_model:
                question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
                
                for subject in subjects_to_search:
                    if subject not in self.datasets['mmlu']:
                        continue
                    
                    for example in self.datasets['mmlu'][subject]:
                        example_text = example['question']
                        example_embedding = self.semantic_model.encode(example_text, convert_to_tensor=True)
                        similarity = util.cos_sim(question_embedding, example_embedding).item()
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_match = example
            
            if best_match and best_score > self.SIMILARITY_THRESHOLD:
                print(f"✅ Found in MMLU (similarity: {best_score:.2%})")
                
                choices = best_match.get('choices', [])
                answer_idx = best_match.get('answer', 0)
                correct_answer = choices[answer_idx] if answer_idx < len(choices) else ""
                
                formatted_choices = "\n".join([f"  {chr(65+i)}. {choice}" 
                                              for i, choice in enumerate(choices)])
                
                answer = f"**Question:** {best_match['question']}\n\n**Choices:**\n{formatted_choices}\n\n**Answer:** {chr(65+answer_idx)}. {correct_answer}"
                
                return {
                    'answer': answer,
                    'source': 'MMLU',
                    'confidence': best_score,
                    'matched_question': best_match['question']
                }
            
            return None
            
        except Exception as e:
            print(f"MMLU error: {e}")
            return None
    
    def detect_subject(self, question):
        """Detect STEM subject from question"""
        question_lower = question.lower()
        scores = {subject: 0 for subject in self.subject_keywords}
        
        for subject, keywords in self.subject_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    scores[subject] += 1
        
        max_score = max(scores.values())
        if max_score == 0:
            return None
        
        return max(scores, key=scores.get)
    
    def format_answer_with_steps(self, answer, topic=None):
        """Format answer with better structure"""
        if topic and any(x in topic.lower() for x in ['equation', 'formula', 'theorem', 'law']):
            # Add visual separators for formulas
            answer = answer.replace('**Formula:**', '\n' + '='*50 + '\n**📐 Formula:**')
            answer = answer.replace('**Example:**', '\n' + '-'*50 + '\n**📝 Example:**')
        
        return answer
    
    def chat(self, question):
        """Enhanced chat with multiple strategies"""
        print(f"\n{'='*70}")
        print(f"💬 Question: {question}")
        print('='*70)
        
        # Validate
        if not question or len(question.strip()) < 3:
            return {'answer': "⚠️ Please ask a complete question.", 'confidence': 0}
        
        # Add to history
        self.conversation_history.append(question)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
        
        # Detect subject
        subject = self.detect_subject(question)
        if subject:
            print(f"📚 Detected subject: {subject.upper()}")
            self.last_subject = subject
        
        # STRATEGY 1: Semantic search in local KB (fastest, most relevant)
        print("\n🔍 Strategy 1: Searching local knowledge base...")
        if self.semantic_model:
            kb_subject, kb_topic, kb_score = self.semantic_search_kb(question)
            
            if kb_score > self.SIMILARITY_THRESHOLD:
                print(f"✅ Found in KB: {kb_topic} (confidence: {kb_score:.2%})")
                content = self.knowledge_base[kb_subject][kb_topic]['content']
                formatted_answer = self.format_answer_with_steps(content, kb_topic)
                
                return {
                    'answer': formatted_answer,
                    'subject': kb_subject,
                    'source': 'Local Knowledge Base',
                    'confidence': kb_score,
                    'topic': kb_topic
                }
        
        # STRATEGY 2: Search ScienceQA (best explanations)
        print("\n🔍 Strategy 2: Searching ScienceQA dataset...")
        scienceqa_result = self.search_scienceqa(question)
        
        if scienceqa_result and scienceqa_result.get('confidence', 0) > self.SIMILARITY_THRESHOLD:
            return scienceqa_result
        
        # STRATEGY 3: Search MMLU (high school specific)
        print("\n🔍 Strategy 3: Searching MMLU dataset...")
        mmlu_result = self.search_mmlu(question, subject)
        
        if mmlu_result and mmlu_result.get('confidence', 0) > self.SIMILARITY_THRESHOLD:
            return mmlu_result
        
        # STRATEGY 4: Search SciQ as final fallback
        if self.datasets.get('sciq'):
            print("\n🔍 Strategy 4: Searching SciQ dataset...")
            sciq_result = self._search_sciq_simple(question)
            if sciq_result:
                return sciq_result
        
        # Return best available match or helpful message
        all_results = [r for r in [scienceqa_result, mmlu_result] if r]
        
        if all_results:
            best_result = max(all_results, key=lambda x: x.get('confidence', 0))
            return {
                'answer': f"⚠️ **Related information** (not an exact match):\n\n{best_result['answer']}",
                'subject': subject,
                'source': best_result['source'] + ' (related)',
                'confidence': best_result.get('confidence', 0) * 0.7,
                'matched_question': best_result.get('matched_question')
            }
        
        # No matches found
        suggestions = self._get_subject_suggestions(subject)
        return {
            'answer': f"""I couldn't find specific information on that topic. 

**Try asking about:**
{suggestions}

**Tips for better results:**
• Be specific (e.g., "What is Newton's first law?" instead of "Tell me about physics")
• Use standard terminology
• Break complex questions into smaller parts""",
            'subject': subject,
            'source': None,
            'confidence': 0.0
        }
    
    def _search_sciq_simple(self, question):
        """Simple SciQ search"""
        try:
            if self.semantic_model:
                question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
                best_match = None
                best_score = 0
                
                for i, example in enumerate(self.datasets['sciq']):
                    if i > 2000:
                        break
                    
                    example_embedding = self.semantic_model.encode(example['question'], convert_to_tensor=True)
                    similarity = util.cos_sim(question_embedding, example_embedding).item()
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = example
                
                if best_match and best_score > self.SIMILARITY_THRESHOLD:
                    print(f"✅ Found in SciQ (similarity: {best_score:.2%})")
                    return {
                        'answer': best_match['correct_answer'],
                        'source': 'SciQ',
                        'confidence': best_score,
                        'matched_question': best_match['question']
                    }
        except:
            pass
        return None
    
    def _get_subject_suggestions(self, subject):
        """Get helpful suggestions based on subject"""
        suggestions = {
            'physics': """• Newton's laws of motion
• Kinetic and potential energy
• Gravity and weight
• Force, mass, and acceleration
• Work and power""",
            'chemistry': """• Atomic structure
• Chemical bonding (ionic, covalent)
• pH and acids/bases
• Chemical reactions
• The periodic table""",
            'biology': """• Photosynthesis
• Cellular respiration
• DNA structure
• Mitosis and meiosis
• Cell structure""",
            'math': """• Quadratic equations
• Pythagorean theorem
• Linear functions
• Trigonometry basics
• Algebra fundamentals"""
        }
        
        if subject and subject in suggestions:
            return suggestions[subject]
        
        return """• Physics: motion, energy, forces
• Chemistry: atoms, bonding, reactions
• Biology: cells, DNA, photosynthesis
• Math: algebra, geometry, calculus"""
    
    def get_practice_problems(self, topic):
        """Suggest practice problems for a topic"""
        # This could be expanded to pull from datasets
        return f"💡 Want practice problems on {topic}? Try searching for specific examples!"
    
    def show_conversation_context(self):
        """Display recent conversation history"""
        if not self.conversation_history:
            return "No conversation history yet."
        
        return "Recent questions:\n" + "\n".join(f"  {i+1}. {q}" 
                                                   for i, q in enumerate(self.conversation_history[-3:]))

def main():
    """Main function to run the enhanced chatbot"""
    print("\n" + "="*70)
    print("  🎓 ENHANCED HIGH SCHOOL STEM TUTOR BOT")
    print("="*70)
    print("\n✨ Features:")
    print("  • Multiple AI-powered datasets")
    print("  • Semantic search for better matching")
    print("  • Step-by-step explanations")
    print("  • Comprehensive high school curriculum coverage")
    print("\nInitializing...\n")
    
    try:
        bot = EnhancedSTEMTutorBot()
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("Make sure you have installed: transformers torch datasets sentence-transformers")
        return
    
    print("\n" + "="*70)
    print("✅ Ready! Ask me anything about high school STEM topics")
    print("="*70)
    
    print("\n💡 **Sample Questions:**")
    print("  • What is Newton's second law?")
    print("  • Explain covalent bonding")
    print("  • What is photosynthesis?")
    print("  • How do I solve quadratic equations?")
    print("  • What is kinetic energy?")
    print("\n📝 Commands:")
    print("  • Type 'history' to see recent questions")
    print("  • Type 'quit' or 'exit' to end")
    print("="*70 + "\n")
    
    while True:
        try:
            question = input("💬 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n👋 Thanks for learning with STEM Tutor Bot!")
                print("Keep studying and stay curious! 🚀\n")
                break
            
            if question.lower() == 'history':
                print(f"\n{bot.show_conversation_context()}\n")
                continue
            
            if not question:
                continue
            
            # Get answer
            result = bot.chat(question)
            
            # Display answer
            print(f"\n{'='*70}")
            print(f"🤖 **Answer:**")
            print(f"{'='*70}")
            print(f"\n{result['answer']}\n")
            
            # Display metadata
            print(f"{'-'*70}")
            
            if result.get('matched_question'):
                print(f"📋 Similar question: {result['matched_question']}")
            
            if result.get('topic'):
                print(f"📖 Topic: {result['topic']}")
            
            if result.get('subject'):
                print(f"📚 Subject: {result['subject'].upper()}")
            
            if result.get('source'):
                source_emoji = "🧠" if 'Local' in result['source'] else "📚"
                print(f"{source_emoji} Source: {result['source']}")
            
            if result.get('confidence', 0) > 0:
                confidence = result['confidence']
                confidence_bar = "█" * int(confidence * 20)
                print(f"🎯 Confidence: {confidence_bar} {confidence:.1%}")
                
                if confidence >= 0.8:
                    print("   ✅ High confidence match!")
                elif confidence >= 0.5:
                    print("   ⚠️ Moderate confidence - answer may be approximate")
                else:
                    print("   ⚠️ Low confidence - consider rephrasing your question")
            
            print(f"{'='*70}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thanks for learning! Goodbye!\n")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.\n")
            continue

if __name__ == "__main__":
    main()
                    