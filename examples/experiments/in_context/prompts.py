##############Prompt Examples##############
student_solution_1 = """Natalie sold half as many clips in April as in May. Then she sold <<48*2=96>> clips in May. \
Then the number of clips in April plus the number of clips in May is <<48+96=48>>. \
Therefore the answer is 48."""

rubric_1 = """1. Points for trying to find the number of clips in May: 2
2. Points for computing <<48/2=24>> clips in May: 1
3. Points for adding the clips in April and May: 1
4. Points for correctly computing <<48+24=72>> total clips: 1"""

rubric_feedback_1 = """1. Natalie sold half as many clips in April as in May: 2/2
2. <<48*2=96>>: 0/1
3. Then the number of clips in April plus the number of clips in May: 1/1
4. <<48+96=48>>: 0/1"""

#####

student_solution_2 = """How much does Weng earn per minute? She earns <<12*60=720>> a minute.
Then Weng earns 720 in total."""

rubric_2 = """1. Points for trying to find earnings per minute: 2
2. Points for computing <<12/60=0.2>> earnings per minute: 1
3. Points for multiplying number of minutes by earnings per minute: 1
4. Points for computing <<0.2*50=10>>: 1
"""

###Maybe only critique first mistake
critique_feedback_1 = """Computing half the number of clips should be done using /, not *."""

refinement_1 = """Natalie sold half as many clips in April as in May. Then she sold <<48/2=24>> clips in May. \
Then the number of clips in April plus the number of clips in May is <<48+24=72>>."""

critique_feedback_2 = """Computing earnings per minute should use /, not *."""

##############Prompt Formats##############
zero_shot_feedback_prompt = """\
Problem:
    {}
Reference solution:
    {}
Student solution:
    {}
Feedback: 
"""

few_shot_feedback_prompt = """\
Problem:
    {}
Reference solution:
    {}
Student Solution:
    {}
Feedback:
    {}

Problem:
    {}
Reference solution:
    {}
Student solution:
    {}
Feedback:
    
"""

#####

few_shot_rubric_prompt = """\
Problem:
    {}
Rubric:
    {}
Student solution:
    {}
Rubric score:
    {}

Problem:
    {}
Rubric:
    {}
Student solution:
    {}
Rubric score:
    
"""

#####

few_shot_rubric_design_prompt = """\
Problem: {}
Solution: {}
Rubric: {}

Problem: {}
Solution: {}
Rubric: 
"""

#####

few_shot_refinement_prompt = """\
Problem: {}
Attempt: {}
Critique: {}
Refinement: {}

Problem: {}
Attempt: {}
Critique: {}
Refinement: """

few_shot_simple_refinement_prompt = """\
Attempt: {}
Critique: {}
Refinement: {}

Attempt: {}
Critique: {}
Refinement: """

few_shot_sentence_refinement_prompt = """\
Sentence: {}
Critique: {}
Refinement: {}

Sentence: {}
Critique: {}
Refinement: """

##############Models##############
#/private/home/alexdahoas/repos/ckpts/llama_7B
#oobabooga/llama-tokenizer
#GeorgiaTechResearchInstitute/galpaca-6.7b
#OpenAssistant/pythia-12b-sft-v8-2.5k-steps
#lmsys/vicuna-13b-delta-v1.1
