Traceback (most recent call last):
  File "sanity_check.py", line 235, in <module>
    main()
  File "sanity_check.py", line 229, in main
    question_1f_sanity_check(model, src_sents, tgt_sents, vocab)
  File "sanity_check.py", line 176, in question_1f_sanity_check
    dec_state_pred, o_t_pred, e_t_pred= model.step(Ybar_t, dec_init_state, enc_hiddens, enc_hiddens_proj, enc_masks)
  File "/Users/zty/Documents/Courses/CS224n/a4/nmt_model.py", line 334, in step
    dec_hidden, dec_cell = self.decoder(Ybar_t, dec_state)
  File "/anaconda3/envs/local_nmt/lib/python3.5/site-packages/torch/nn/modules/module.py", line 489, in __call__
    result = self.forward(*input, **kwargs)
  File "/anaconda3/envs/local_nmt/lib/python3.5/site-packages/torch/nn/modules/rnn.py", line 723, in forward
    self.check_forward_input(input)
  File "/anaconda3/envs/local_nmt/lib/python3.5/site-packages/torch/nn/modules/rnn.py", line 566, in check_forward_input
    input.size(1), self.input_size))
RuntimeError: input has inconsistent input_size: got 6, expected 3
