from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from gensim.models import KeyedVectors
import math
from django.core.files.storage import FileSystemStorage
from urllib.parse import urlencode

from .models import Question, Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *

import os
current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.
def index(request):
    questions_list = Question.objects.order_by('set')
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    question = get_object_or_404(Question, pk=question_id)
    context = {
        "essay": essay,
        "question": question,
    }
    return render(request, 'grader/essay.html', context)

def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST, request.FILES)
        if form.is_valid():

            content = form.cleaned_data.get('answer')
            myfile = request.FILES['file']
            
            if len(content) > 20:
                num_features = 300
                model = KeyedVectors.load_word2vec_format(os.path.join(current_path, "deep_learning_files/word2vecmodel.bin"), binary=True)
                #print(model)
                clean_essay = []
                clean_essay.append(essay_to_wordlist( content, remove_stopwords=True ))
                testDataVecs = getAvgFeatureVecs( clean_essay, model, num_features )
                testDataVecs = np.array(testDataVecs)
                testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

                lstm_model = get_model()
                lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/optimized_lstm.h5"))
                essay_score = float(lstm_model.predict(testDataVecs))

                if math.isnan(essay_score) or essay_score < 0:
                    essay_score = 0
                
                else:
                    essay_score = np.around(essay_score*int(question.max_score)/10)

                fs = FileSystemStorage()
                fs.save(myfile.name, myfile)
                keywords = handle_uploaded_file(myfile) 
                #print(keywords)

                keyword_score = calculate_score(clean_essay, keywords)
                keyword_score = np.around(keyword_score*int(question.max_score)/10)
                #print(essay_score, keyword_score)
                final_score = (essay_score + keyword_score)/2
            else:   
                final_score = 0
            K.clear_session()
            essay = Essay.objects.create(
                content=content,
                question=question,
                score=final_score
            )

            # Store the required values in session variables
            request.session['essay_score'] = essay_score
            request.session['keyword_score'] = keyword_score
            request.session['clean_essay'] = content

        return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "question": question,
        "form": form,
    }
    return render(request, 'grader/question.html', context)

def essay_detail(request, question_id, essay_id):
    essay = get_object_or_404(Essay, id=essay_id)
    question = essay.question 
    # Retrieve the required values from session variables
    essay_score = request.session.get('essay_score', 0.0)
    keyword_score = request.session.get('keyword_score', 0.0)
    clean_essay = request.session.get('clean_essay', '')
    del request.session['essay_score']
    del request.session['keyword_score']
    del request.session['clean_essay']

    # Perform your other calculations and logic here
    # ...
    grammatical_errors = count_grammatical_errors(clean_essay)
    spelling_errors = count_spelling_errors(clean_essay)
    redundancies = count_redundancies(clean_essay)
    essay_length = no_of_words(clean_essay)
    # Calculate the grammatical_errors, spelling_errors, redundancies, and essay_length
    # ...


    context = {
        'essay': essay,
        'essay_score': essay_score,
        'keyword_based_score': keyword_score,
        'grammatical_errors': grammatical_errors,
        'spelling_errors': spelling_errors,
        'redundancies': redundancies,
        'essay_length': essay_length,
        'question': question,
        'question_id': question_id,
    }

    return render(request, 'grader/essay_detail.html', context)
