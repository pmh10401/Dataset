alter table public.operational_alpha_settings
  add column if not exists last_error text
  check (last_error is null or char_length(last_error) <= 500);

create or replace function public.acknowledge_operational_alpha_change(
  p_source_id text,
  p_version bigint,
  p_alpha numeric,
  p_success boolean,
  p_error_message text default null
)
returns public.operational_alpha_settings
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid := auth.uid();
  v_setting public.operational_alpha_settings;
begin
  if v_user_id is null then
    raise exception 'authentication required' using errcode = '42501';
  end if;
  if not exists (
    select 1 from public.dashboard_members m
    where m.user_id = v_user_id and m.can_write
  ) then
    raise exception 'write permission required' using errcode = '42501';
  end if;
  if p_source_id is null or char_length(btrim(p_source_id)) not between 1 and 120 then
    raise exception 'invalid source id' using errcode = '22023';
  end if;
  if p_version is null or p_version < 1 then
    raise exception 'invalid version' using errcode = '22023';
  end if;
  if p_alpha not in (0.005, 0.010) then
    raise exception 'invalid alpha preset' using errcode = '22023';
  end if;
  if p_error_message is not null and char_length(p_error_message) > 500 then
    raise exception 'error message is too long' using errcode = '22023';
  end if;

  update public.operational_alpha_settings
  set effective_alpha = case when p_success then p_alpha else effective_alpha end,
      status = case when p_success then 'applied' else 'failed' end,
      applied_at = case when p_success then now() else null end,
      last_error = case when p_success then null else coalesce(p_error_message, 'unknown apply failure') end,
      updated_at = now()
  where source_id = btrim(p_source_id)
    and version = p_version
    and desired_alpha = p_alpha
    and status = 'pending'
  returning * into v_setting;

  if v_setting.source_id is null then
    raise exception 'stale or mismatched alpha acknowledgement' using errcode = '40001';
  end if;

  insert into public.operational_alpha_setting_events (
    source_id,
    previous_desired_alpha,
    requested_alpha,
    status,
    requested_by,
    reason
  )
  values (
    v_setting.source_id,
    v_setting.desired_alpha,
    p_alpha,
    case when p_success then 'applied' else 'failed' end,
    v_user_id,
    case
      when p_success then 'Mac detector acknowledged local Alpha application'
      else left(
        'Mac detector rejected Alpha application: ' || coalesce(p_error_message, 'unknown failure'),
        500
      )
    end
  );

  return v_setting;
end;
$$;

revoke all on function public.acknowledge_operational_alpha_change(
  text, bigint, numeric, boolean, text
) from public;
grant execute on function public.acknowledge_operational_alpha_change(
  text, bigint, numeric, boolean, text
) to authenticated;
